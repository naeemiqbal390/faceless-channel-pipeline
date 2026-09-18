"""
app.py
Streamlit Web Application UI
"""

import os
import re
import csv
import json
import uuid
import hashlib
import tempfile
import asyncio
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd


def get_session_temp_dir():
    """
    Every generated file (narration.mp3, scene_manifest.csv, final_video.mp4,
    ...) used a fixed shared filename under the OS-wide temp dir. That's fine
    for one person in one tab, but two concurrent sessions (two browser tabs,
    or two people) would silently overwrite each other's files mid-run. This
    gives each session its own subfolder instead.
    """
    if "session_temp_dir" not in st.session_state:
        session_dir = os.path.join(tempfile.gettempdir(), "session_" + uuid.uuid4().hex[:12])
        os.makedirs(session_dir, exist_ok=True)
        st.session_state["session_temp_dir"] = session_dir
    return st.session_state["session_temp_dir"]


def compute_project_id(script_text):
    """
    A stable ID derived from the script's own content — the same script
    pasted back in later (even after a full container restart, with zero
    other state carried over) resolves to the same project automatically,
    with no separate bookkeeping needed.
    """
    return hashlib.sha256(script_text.encode("utf-8")).hexdigest()[:16]


def get_project_status(project_id):
    """
    Returns what's saved in B2 for this project, or None if B2 isn't
    configured or nothing's saved yet. This is what /tmp being wiped on
    every container restart/sleep/redeploy can't take away — it lives
    outside the app's ephemeral filesystem entirely.
    """
    import b2_storage
    if not b2_storage.is_configured():
        return None
    prefix = f"projects/{project_id}/"
    keys = b2_storage.list_keys(prefix)
    if not keys:
        return None
    image_keys = [k for k in keys if "/scene_images/" in k and k.endswith(".png")]
    return {
        "has_audio": any(k.endswith("narration.mp3") for k in keys),
        "has_manifest": any(k.endswith("scene_manifest.csv") for k in keys),
        "num_images": len(image_keys),
        "has_video": any(k.endswith("final_video.mp4") for k in keys),
    }


def resume_project_from_b2(project_id):
    """Downloads every saved artifact for this project into the current
    session's local workspace and restores session_state to match, so the
    rest of the app picks up exactly where it left off — including image
    generation, since pollinations_runner's own skip-if-exists resume logic
    then sees these files as already done."""
    import b2_storage
    session_dir = get_session_temp_dir()
    prefix = f"projects/{project_id}/"

    audio_key = prefix + "narration.mp3"
    if b2_storage.key_exists(audio_key):
        local_audio = os.path.join(session_dir, "narration.mp3")
        if b2_storage.download_file(audio_key, local_audio):
            st.session_state["audio_path"] = local_audio

    manifest_key = prefix + "scene_manifest.csv"
    local_manifest = None
    if b2_storage.key_exists(manifest_key):
        local_manifest = os.path.join(session_dir, "scene_manifest.csv")
        if b2_storage.download_file(manifest_key, local_manifest):
            st.session_state["manifest_path"] = local_manifest

    if local_manifest:
        import pollinations_runner
        images_dir = pollinations_runner.get_images_dir(local_manifest)
        image_keys = [k for k in b2_storage.list_keys(prefix + "scene_images/") if k.endswith(".png")]
        for k in image_keys:
            b2_storage.download_file(k, os.path.join(images_dir, os.path.basename(k)))
        if image_keys:
            st.session_state["images_dir"] = images_dir

    video_key = prefix + "final_video.mp4"
    if b2_storage.key_exists(video_key):
        local_video = os.path.join(session_dir, "final_video.mp4")
        if b2_storage.download_file(video_key, local_video):
            st.session_state["video_path"] = local_video

    st.session_state["project_id"] = project_id


def upload_to_project(local_path, relative_name):
    """
    Best-effort upload of a finished artifact to B2 under the current
    script's project ID. Never raises — persistence failing should never
    break the pipeline itself, just mean that piece isn't backed up.
    """
    script_text = st.session_state.get("script_text", "")
    if not script_text or not local_path or not os.path.exists(local_path):
        return
    try:
        import b2_storage
        if not b2_storage.is_configured():
            return
        project_id = compute_project_id(script_text)
        b2_storage.upload_file(local_path, f"projects/{project_id}/{relative_name}")
    except Exception:
        pass


def force_fix_manifest_csv(csv_path):
    """Guarantees schema normalization directly prior to triggering background execution."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path)

    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "").str.lower()

    rename_dict = {}
    for col in df.columns:
        if col in ["scene", "scene id", "scene_number", "id", "sn", "unnamed: 0"]:
            rename_dict[col] = "scene_id"
        elif col in ["prompts", "image_prompt", "scene_prompt", "description", "text"]:
            rename_dict[col] = "prompt"

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    if "scene_id" not in df.columns:
        df["scene_id"] = list(range(1, len(df) + 1))

    if "prompt" not in df.columns:
        df["prompt"] = "stick figure drawing"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


EDGE_TTS_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
]


def get_secret(key_names, user_input=""):
    """
    key_names can be a single string or a list of acceptable names — tries
    each in order. This exists because a secret named slightly differently
    than what the code expects (e.g. POLLINATIONS_API_KEY vs
    POLLINATIONS_TOKEN) fails a plain st.secrets[name] lookup silently,
    with no error — it just quietly returns empty, which is exactly what
    happened here.
    """
    if user_input and user_input.strip():
        return user_input.strip()
    names = key_names if isinstance(key_names, (list, tuple)) else [key_names]
    for name in names:
        try:
            value = st.secrets[name]
            if value:
                return value
        except Exception:
            continue
    return ""


def parse_script_scenes(raw):
    pattern = re.compile(r"\[SCENE\s+\d+:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(raw))
    scenes = []
    for i, m in enumerate(matches):
        prompt = m.group(1).strip().replace("\n", " ")
        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        narration = raw[start_pos:end_pos].strip()
        scenes.append({"scene_id": i + 1, "prompt": prompt, "narration": narration})
    return scenes


def generate_audio_file(script_text, voice, progress_callback=None, scenes_per_chunk=7, work_dir=None):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found — check script input.")

    work_dir = work_dir or tempfile.gettempdir()
    chunks = [scenes[i:i + scenes_per_chunk] for i in range(0, len(scenes), scenes_per_chunk)]
    total_chunks = len(chunks)

    import edge_tts

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_text = " ".join(s["narration"] for s in chunk if s["narration"])
        if not chunk_text.strip():
            continue
        chunk_path = os.path.join(work_dir, f"narration_chunk_{i}.mp3")

        async def _gen(text=chunk_text, path=chunk_path):
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(path)

        asyncio.run(_gen())
        chunk_paths.append(chunk_path)

        if progress_callback:
            progress_callback(i + 1, total_chunks)

    out_path = os.path.join(work_dir, "narration.mp3")
    with open(out_path, "wb") as outfile:
        for cp in chunk_paths:
            with open(cp, "rb") as infile:
                outfile.write(infile.read())

    return out_path


def align_script_to_audio_file(script_text, audio_path, work_dir=None):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found in script.")

    work_dir = work_dir or tempfile.gettempdir()

    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    whisper_words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                whisper_words.append({"word": w.word.strip().lower(), "start": w.start, "end": w.end})

    def normalize(text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    plain_words = [w["word"] for w in whisper_words]
    cursor = 0
    for scene in scenes:
        target_words = normalize(scene["narration"])
        if not target_words or not whisper_words:
            scene["start_time"], scene["end_time"] = None, None
            continue

        window_end = min(len(plain_words), cursor + len(target_words) * 3 + 20)
        window = plain_words[cursor:window_end]
        matcher = SequenceMatcher(None, window, target_words)
        match = matcher.find_longest_match(0, len(window), 0, len(target_words))

        if match.size == 0:
            span_len = min(len(target_words), len(window)) or 1
            start_idx = cursor
            end_idx = min(cursor + span_len - 1, len(whisper_words) - 1)
        else:
            start_idx = cursor + match.a
            approx_span = max(match.size, len(target_words) - match.b)
            end_idx = min(start_idx + approx_span - 1, len(whisper_words) - 1)

        start_idx = max(0, min(start_idx, len(whisper_words) - 1))
        end_idx = max(start_idx, min(end_idx, len(whisper_words) - 1))

        scene["start_time"] = whisper_words[start_idx]["start"]
        scene["end_time"] = whisper_words[end_idx]["end"]
        cursor = end_idx + 1

    def seconds_to_hhmmss(seconds):
        if seconds is None:
            return ""
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

    out_path = os.path.join(work_dir, "scene_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "start_time", "end_time", "prompt"])
        for s in scenes:
            writer.writerow([s["scene_id"], seconds_to_hhmmss(s["start_time"]),
                              seconds_to_hhmmss(s["end_time"]), s["prompt"]])
    return out_path, len(scenes)


def make_progress_log(container, max_lines_shown=40):
    """
    Accumulates progress messages into a scrollable log instead of
    overwriting the previous one — the earlier version only ever showed
    the single latest line, so a scene that failed 5 messages ago (and
    the reason it failed) was already gone by the time anything else
    printed. Every line is also printed to stdout, so the full history
    (not just the last 40 shown on screen) is always recoverable from the
    Streamlit Cloud app's own logs even after the browser tab is closed.
    """
    lines = []

    def _log(message):
        lines.append(message)
        print(message, flush=True)
        container.code("\n".join(lines[-max_lines_shown:]), language=None)

    return _log


def make_image_gallery(container, images_dir, cols=5, max_shown=15):
    """
    Renders the most recently generated images as a thumbnail grid,
    refreshed on demand — lets you actually see quality along the way
    instead of waiting for the whole batch to finish. Reads the images
    directory fresh each call rather than tracking state, since
    pollinations_runner writes deterministic scene_NNN.png filenames.
    """
    def _refresh():
        try:
            files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))
        except Exception:
            files = []
        recent = files[-max_shown:]
        with container.container():
            if not recent:
                st.caption("Images will appear here as they're generated.")
                return
            rows = [recent[i:i + cols] for i in range(0, len(recent), cols)]
            for row in rows:
                row_cols = st.columns(cols)
                for cell, fname in zip(row_cols, row):
                    cell.image(os.path.join(images_dir, fname), use_container_width=True,
                               caption=fname.replace(".png", ""))

    return _refresh


# Interface setup
st.set_page_config(page_title="Faceless Channel Pipeline", layout="wide")
st.title("Faceless Channel Pipeline")
st.caption("Stick-figure style · unlimited characters · zero cost")

if "manifest_path" not in st.session_state:
    st.session_state["manifest_path"] = None

tab0, tab_audio, tab_align, tab_images, tab_video = st.tabs(
    ["1. Paste script", "2. Audio", "3. Align", "4. Images", "5. Video"]
)

with tab0:
    st.markdown("Paste the finished script here to start the automated workflow.")
    pasted_script = st.text_area("Script", st.session_state.get("script_text", ""), height=400)
    col_a, col_b = st.columns(2)
    if col_a.button("Save script", type="primary"):
        if "[SCENE" not in pasted_script:
            st.warning("No [SCENE N: ...] markers found in script input.")
        else:
            st.session_state["script_text"] = pasted_script
            scene_count = len(re.findall(r"\[SCENE", pasted_script))
            narration_only = re.sub(r"\[SCENE.*?\]\n?", "", pasted_script, flags=re.DOTALL)
            word_count = len(re.findall(r"\S+", narration_only))
            st.success(f"Saved — {scene_count} scenes, ~{word_count} words.")

    # Check for previously saved progress on this exact script — survives
    # container restarts, sleeps, and walking away for hours, since it's
    # read from B2 rather than this session's (ephemeral) state.
    saved_script_check = st.session_state.get("script_text", "")
    if saved_script_check:
        import b2_storage
        if b2_storage.is_configured():
            check_project_id = compute_project_id(saved_script_check)
            project_status = get_project_status(check_project_id)
            if project_status:
                status_bits = [
                    "audio ✓" if project_status["has_audio"] else "audio ✗",
                    "manifest ✓" if project_status["has_manifest"] else "manifest ✗",
                    f"{project_status['num_images']} images saved",
                    "video ✓" if project_status["has_video"] else "video ✗",
                ]
                st.info(f"Saved progress found for this script: {', '.join(status_bits)}")
                resume_col, clear_col = st.columns(2)
                if resume_col.button("Resume — load saved progress", type="primary"):
                    with st.spinner("Downloading saved progress from B2..."):
                        resume_project_from_b2(check_project_id)
                    st.success("Progress restored — shown below.")
                if clear_col.button("Clear saved progress — back to stage 1"):
                    b2_storage.delete_prefix(f"projects/{check_project_id}/")
                    for key in ["script_text", "audio_path", "manifest_path", "images_dir",
                                "video_path", "project_id"]:
                        st.session_state.pop(key, None)
                    st.success("Cleared. Starting over.")
                    st.rerun()
        else:
            st.caption("Add B2_KEY_ID / B2_APPLICATION_KEY / B2_BUCKET_NAME / B2_ENDPOINT_URL to Secrets to "
                       "enable save/resume across sessions and restarts.")

    # Shown right here, in the same tab, whenever anything is loaded —
    # right after Resume, or after any pipeline step completes — so there's
    # no need to go hunting through other tabs to confirm something worked.
    has_any_status = any(st.session_state.get(k) for k in
                          ["audio_path", "manifest_path", "images_dir", "video_path"])
    if has_any_status:
        st.markdown("**Current project status:**")
        audio_p = st.session_state.get("audio_path")
        if audio_p and os.path.exists(audio_p):
            st.audio(audio_p)
        manifest_p = st.session_state.get("manifest_path")
        if manifest_p and os.path.exists(manifest_p):
            st.caption(f"Manifest ready: {os.path.basename(manifest_p)}")
        images_d = st.session_state.get("images_dir")
        if images_d and os.path.isdir(images_d):
            n_imgs = len([f for f in os.listdir(images_d) if f.lower().endswith(".png")])
            st.caption(f"{n_imgs} image(s) available.")
        video_p = st.session_state.get("video_path")
        if video_p and os.path.exists(video_p):
            st.video(video_p)
            with open(video_p, "rb") as f:
                st.download_button("Download final_video.mp4", f, file_name="final_video.mp4",
                                    mime="video/mp4", key="dl_video_tab0_status")

    st.divider()
    voice_auto = st.selectbox("Voice", EDGE_TTS_VOICES, key="voice_auto")

    import pollinations_runner
    style_options = list(pollinations_runner.STYLE_PRESETS.keys())
    style_auto = st.selectbox("Image style", style_options, key="style_auto")
    st.session_state["image_style"] = style_auto

    pollinations_token_auto = st.text_input(
        "Pollinations token (optional — leave blank to use anonymously, or set POLLINATIONS_TOKEN in Secrets)",
        type="password", key="pk_auto",
    )

    if col_b.button("Run full pipeline", type="primary"):
        if "[SCENE" not in pasted_script:
            st.warning("No [SCENE N: ...] markers found.")
        else:
            st.session_state["script_text"] = pasted_script
            try:
                st.markdown("**Step 1/4 — audio**")
                audio_bar = st.progress(0.0)
                audio_status = st.empty()

                def on_audio_progress(done, total):
                    audio_bar.progress(done / total if total else 0)
                    audio_status.markdown(f"{done}/{total} audio segments")

                audio_path = generate_audio_file(pasted_script, voice_auto, progress_callback=on_audio_progress,
                                                  work_dir=get_session_temp_dir())
                st.session_state["audio_path"] = audio_path
                upload_to_project(audio_path, "narration.mp3")
                audio_bar.progress(1.0)
                with open(audio_path, "rb") as f:
                    st.download_button("Download narration.mp3", f, file_name="narration.mp3",
                                        mime="audio/mpeg", key="dl_audio_pipeline")

                with st.spinner("Step 2/4 — aligning..."):
                    manifest_path, n_scenes = align_script_to_audio_file(pasted_script, audio_path,
                                                                          work_dir=get_session_temp_dir())
                    # Normalize BEFORE saving anywhere — get_images_dir hashes the
                    # file's bytes, and normalizing later would change those
                    # bytes, making the B2-saved copy hash differently than the
                    # copy actually used for image generation.
                    force_fix_manifest_csv(manifest_path)
                    st.session_state["manifest_path"] = manifest_path
                    upload_to_project(manifest_path, "scene_manifest.csv")
                with open(manifest_path, "rb") as f:
                    st.download_button("Download scene_manifest.csv", f, file_name="scene_manifest.csv",
                                        mime="text/csv", key="dl_manifest_pipeline")

                st.markdown("**Step 3/4 — images**")
                import pollinations_runner
                pollinations_token = get_secret(["POLLINATIONS_TOKEN", "POLLINATIONS_API_KEY"], pollinations_token_auto)

                active_manifest = st.session_state.get("manifest_path")
                if not active_manifest or not os.path.exists(active_manifest):
                    raise ValueError("Manifest path unresolved or non-existent.")

                # Must normalize BEFORE computing the images directory — it's
                # hashed from the file's bytes, and normalizing changes those
                # bytes (line endings/encoding) even for an already-valid
                # file. Doing this after previewing the gallery directory
                # made the preview point at a different folder than the one
                # generation actually wrote to.
                force_fix_manifest_csv(active_manifest)
                images_dir_preview = pollinations_runner.get_images_dir(active_manifest)

                log_col, gallery_col = st.columns([2, 3])
                with log_col:
                    image_bar = st.progress(0.0)
                    image_status = st.empty()
                image_log = make_progress_log(image_status)
                with gallery_col:
                    st.caption("Generated so far (most recent):")
                    gallery_placeholder = st.empty()
                refresh_gallery = make_image_gallery(gallery_placeholder, images_dir_preview)
                refresh_gallery()

                def on_image_progress(done_images, total_images, message):
                    pct = done_images / total_images if total_images else 0
                    image_bar.progress(min(pct, 1.0))
                    image_log(f"{done_images}/{total_images} images generated — {message}")
                    refresh_gallery()

                def on_image_saved(scene_id, local_path):
                    upload_to_project(local_path, f"scene_images/{os.path.basename(local_path)}")

                images_dir, zip_path, failed_scenes = pollinations_runner.run_image_generation(
                    active_manifest, pollinations_token, progress_callback=on_image_progress,
                    style=st.session_state.get("image_style", pollinations_runner.DEFAULT_STYLE),
                    on_image_saved=on_image_saved,
                )
                st.session_state["images_dir"] = images_dir
                image_bar.progress(1.0)
                if failed_scenes:
                    st.warning(f"{len(failed_scenes)} scenes failed and were skipped: {failed_scenes}. "
                               f"Re-run image generation to retry just those.")
                else:
                    st.success("All images generated.")
                with open(zip_path, "rb") as f:
                    st.download_button("Download scene_images_batch.zip", f, file_name="scene_images_batch.zip",
                                        mime="application/zip", key="dl_images_pipeline")

                st.markdown("**Step 4/4 — video**")
                video_bar = st.progress(0.0)
                video_status = st.empty()
                video_log = make_progress_log(video_status)

                def on_video_progress(done_scenes, total_scenes, message):
                    pct = done_scenes / total_scenes if total_scenes else 0
                    video_bar.progress(min(pct, 1.0))
                    video_log(f"{done_scenes}/{total_scenes} scenes — {message}")

                import video_assembler
                output_video_path = os.path.join(get_session_temp_dir(), "final_video.mp4")
                final_path, filled_scenes = video_assembler.assemble_video(
                    active_manifest, images_dir, audio_path, output_video_path,
                    progress_callback=on_video_progress,
                )
                st.session_state["video_path"] = final_path
                upload_to_project(final_path, "final_video.mp4")
                video_bar.progress(1.0)
                if filled_scenes:
                    st.warning(f"Video assembled and fully in sync — but {len(filled_scenes)} scene(s) didn't "
                               f"have their own image yet, so a neighboring scene's picture was used to fill "
                               f"that gap: {filled_scenes}. Regenerate those in the Images tab, then reassemble "
                               f"to swap in the real artwork.")
                else:
                    st.success("Video assembled.")

                st.video(final_path)
                with open(final_path, "rb") as f:
                    st.download_button("Download final_video.mp4", f, file_name="final_video.mp4", mime="video/mp4",
                                        key="dl_video_pipeline")

            except Exception as e:
                st.error(f"Pipeline stopped: {e}")

with tab_audio:
    saved_script = st.session_state.get("script_text", "")
    if not saved_script:
        st.warning("No script saved yet — paste and save your script in tab 1 first.")
    else:
        with st.expander("Script being used (edit it in tab 1, not here)"):
            st.text_area("Script", saved_script, height=200, key="audio_script_preview", disabled=True)

    voice = st.selectbox("Voice", EDGE_TTS_VOICES)
    if st.button("Generate audio", type="primary", disabled=not saved_script):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_audio_progress(done, total):
            progress_bar.progress(done / total if total else 0)
            status_text.markdown(f"**{done}/{total} audio segments** generated")

        try:
            path = generate_audio_file(saved_script, voice, progress_callback=on_audio_progress,
                                        work_dir=get_session_temp_dir())
            st.session_state["audio_path"] = path
            upload_to_project(path, "narration.mp3")
            progress_bar.progress(1.0)
            st.success("Audio generated.")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
    if "audio_path" in st.session_state:
        st.audio(st.session_state["audio_path"])
        with open(st.session_state["audio_path"], "rb") as f:
            st.download_button("Download narration.mp3", f, file_name="narration.mp3", mime="audio/mpeg",
                                key="dl_audio_tab")

with tab_align:
    saved_script2 = st.session_state.get("script_text", "")
    if not saved_script2:
        st.warning("No script saved yet — paste and save your script in tab 1 first.")
    else:
        with st.expander("Script being used (edit it in tab 1, not here)"):
            st.text_area("Script", saved_script2, height=200, key="align_script_preview", disabled=True)

    uploaded_audio = st.file_uploader("Finished audio (optional)", type=["mp3", "wav"])
    if st.button("Generate scene manifest", type="primary", disabled=not saved_script2):
        if not uploaded_audio and "audio_path" not in st.session_state:
            st.warning("Generate or upload audio first.")
        else:
            with st.spinner("Aligning..."):
                try:
                    if uploaded_audio:
                        audio_path = os.path.join(get_session_temp_dir(), "uploaded_audio")
                        with open(audio_path, "wb") as f:
                            f.write(uploaded_audio.read())
                        st.session_state["audio_path"] = audio_path
                    else:
                        audio_path = st.session_state["audio_path"]
                    manifest_path, n_scenes = align_script_to_audio_file(saved_script2, audio_path,
                                                                          work_dir=get_session_temp_dir())
                    force_fix_manifest_csv(manifest_path)
                    st.session_state["manifest_path"] = manifest_path
                    upload_to_project(manifest_path, "scene_manifest.csv")
                    st.success(f"Aligned {n_scenes} scenes.")
                    with open(manifest_path, "rb") as f:
                        st.download_button("Download scene_manifest.csv", f, file_name="scene_manifest.csv",
                                            key="dl_manifest_tab")
                except Exception as e:
                    st.error(f"Alignment failed: {e}")

with tab_images:
    st.markdown("### Generate Images (Pollinations.ai)")
    st.caption("Free, keyless image generation (Flux, falling back to Turbo). Paces itself dynamically instead of "
               "a fixed rate — a failed scene is parked and auto-retried later without blocking the rest. "
               "Works anonymously; a free Pollinations account token (no card) raises the rate limit and drops "
               "the watermark, but isn't required.")

    uploaded_manifest = st.file_uploader("scene_manifest.csv (optional)", type=["csv"], key="tab_images_uploader")

    if uploaded_manifest is not None:
        temp_manifest_path = os.path.join(get_session_temp_dir(), "scene_manifest.csv")
        with open(temp_manifest_path, "wb") as f:
            f.write(uploaded_manifest.getvalue())
        st.session_state["manifest_path"] = temp_manifest_path

    active_manifest_path = st.session_state.get("manifest_path")

    if active_manifest_path and os.path.exists(active_manifest_path):
        st.info(f"Loaded manifest ready: {os.path.basename(active_manifest_path)}")
        import pollinations_runner
        try:
            resume_info = pollinations_runner.get_resume_status(active_manifest_path)
            if resume_info.get("has_progress"):
                st.success(f"Found saved progress: {resume_info['done_images']}/{resume_info['total_images']} images already done.")
        except Exception:
            pass
    else:
        st.warning("No active manifest found. Upload or generate a CSV manifest.")

    pollinations_token_override = st.text_input(
        "Pollinations token (optional — or set POLLINATIONS_TOKEN in Secrets; blank works fine anonymously)",
        type="password", key="tab_images_key",
    )

    import pollinations_runner
    style_options = list(pollinations_runner.STYLE_PRESETS.keys())
    default_style = st.session_state.get("image_style", pollinations_runner.DEFAULT_STYLE)
    default_index = style_options.index(default_style) if default_style in style_options else 0
    style_override = st.selectbox("Image style", style_options, index=default_index, key="tab_images_style")
    st.session_state["image_style"] = style_override

    if st.button("Generate Images", type="primary"):
        if not active_manifest_path or not os.path.exists(active_manifest_path):
            st.error("Missing manifest file! Please upload a CSV or run alignment.")
        else:
            try:
                pollinations_token = get_secret(["POLLINATIONS_TOKEN", "POLLINATIONS_API_KEY"], pollinations_token_override)

                import pollinations_runner
                force_fix_manifest_csv(active_manifest_path)
                images_dir_preview = pollinations_runner.get_images_dir(active_manifest_path)

                log_col, gallery_col = st.columns([2, 3])
                with log_col:
                    image_bar = st.progress(0.0)
                    image_status = st.empty()
                image_log = make_progress_log(image_status)
                with gallery_col:
                    st.caption("Generated so far (most recent):")
                    gallery_placeholder = st.empty()
                refresh_gallery = make_image_gallery(gallery_placeholder, images_dir_preview)
                refresh_gallery()

                def on_image_progress(done_images, total_images, message):
                    pct = done_images / total_images if total_images else 0
                    image_bar.progress(min(pct, 1.0))
                    image_log(f"{done_images}/{total_images} images generated — {message}")
                    refresh_gallery()

                def on_image_saved(scene_id, local_path):
                    upload_to_project(local_path, f"scene_images/{os.path.basename(local_path)}")

                images_dir, zip_path, failed_scenes = pollinations_runner.run_image_generation(
                    active_manifest_path, pollinations_token, progress_callback=on_image_progress,
                    style=style_override,
                    on_image_saved=on_image_saved,
                )
                st.session_state["images_dir"] = images_dir

                image_bar.progress(1.0)
                if failed_scenes:
                    st.warning(f"{len(failed_scenes)} scenes failed and were skipped: {failed_scenes}. "
                               f"Click Generate Images again to retry just those — completed scenes won't be redone.")
                else:
                    st.success("All images generated successfully!")

                with open(zip_path, "rb") as f:
                    st.download_button(
                        "Download scene_images_batch.zip",
                        f,
                        file_name="scene_images_batch.zip",
                        mime="application/zip",
                        key="dl_images_tab"
                        )
            except Exception as e:
                st.error(f"Image generation failed: {e}")

with tab_video:
    st.markdown("### Assemble Final Video")
    st.caption("Combines generated scene images + narration audio into one MP4, timed to the scene manifest.")

    active_manifest_for_video = st.session_state.get("manifest_path")
    active_audio_for_video = st.session_state.get("audio_path")

    uploaded_audio_video = None
    if not active_audio_for_video:
        uploaded_audio_video = st.file_uploader("Narration audio (optional, if not already generated)", type=["mp3", "wav"], key="tab_video_audio")
        if uploaded_audio_video is not None:
            active_audio_for_video = os.path.join(get_session_temp_dir(), "uploaded_audio_for_video")
            with open(active_audio_for_video, "wb") as f:
                f.write(uploaded_audio_video.read())
            st.session_state["audio_path"] = active_audio_for_video

    if active_manifest_for_video and os.path.exists(active_manifest_for_video):
        import pollinations_runner
        images_dir_for_video = pollinations_runner.get_images_dir(active_manifest_for_video)
        resume_info = pollinations_runner.get_resume_status(active_manifest_for_video)
        st.info(f"Images ready: {resume_info['done_images']}/{resume_info['total_images']}")
    else:
        images_dir_for_video = None
        st.warning("No active manifest found. Generate/align a script first, or upload a manifest in the Images tab.")

    if not active_audio_for_video:
        st.warning("No narration audio found. Generate audio first, or upload it above.")

    if st.button("Assemble Video", type="primary"):
        if not active_manifest_for_video or not images_dir_for_video:
            st.error("Missing scene manifest.")
        elif not active_audio_for_video or not os.path.exists(active_audio_for_video):
            st.error("Missing narration audio.")
        else:
            try:
                video_bar = st.progress(0.0)
                video_status = st.empty()
                video_log = make_progress_log(video_status)

                def on_video_progress(done_scenes, total_scenes, message):
                    pct = done_scenes / total_scenes if total_scenes else 0
                    video_bar.progress(min(pct, 1.0))
                    video_log(f"{done_scenes}/{total_scenes} scenes — {message}")

                import video_assembler
                output_video_path = os.path.join(get_session_temp_dir(), "final_video.mp4")
                final_path, filled_scenes = video_assembler.assemble_video(
                    active_manifest_for_video, images_dir_for_video, active_audio_for_video,
                    output_video_path, progress_callback=on_video_progress,
                )
                st.session_state["video_path"] = final_path
                upload_to_project(final_path, "final_video.mp4")
                video_bar.progress(1.0)

                if filled_scenes:
                    st.warning(f"Video assembled and fully in sync — but {len(filled_scenes)} scene(s) didn't "
                               f"have their own image yet, so a neighboring scene's picture was used to fill "
                               f"that gap: {filled_scenes}. Regenerate those in the Images tab, then reassemble "
                               f"to swap in the real artwork.")
                else:
                    st.success("Video assembled successfully!")

            except Exception as e:
                st.error(f"Video assembly failed: {e}")

    if st.session_state.get("video_path") and os.path.exists(st.session_state["video_path"]):
        st.video(st.session_state["video_path"])
        with open(st.session_state["video_path"], "rb") as f:
            st.download_button("Download final_video.mp4", f, file_name="final_video.mp4", mime="video/mp4",
                                key="dl_video_tab")

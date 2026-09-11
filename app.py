"""
app.py
Streamlit Web Application UI
"""

import os
import re
import csv
import json
import uuid
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


def get_secret(key_name, user_input=""):
    if user_input and user_input.strip():
        return user_input.strip()
    try:
        return st.secrets[key_name]
    except Exception:
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

    st.divider()
    voice_auto = st.selectbox("Voice", EDGE_TTS_VOICES, key="voice_auto")
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
                audio_bar.progress(1.0)

                with st.spinner("Step 2/4 — aligning..."):
                    manifest_path, n_scenes = align_script_to_audio_file(pasted_script, audio_path,
                                                                          work_dir=get_session_temp_dir())
                    st.session_state["manifest_path"] = manifest_path

                st.markdown("**Step 3/4 — images**")
                image_bar = st.progress(0.0)
                image_status = st.empty()
                image_log = make_progress_log(image_status)

                def on_image_progress(done_images, total_images, message):
                    pct = done_images / total_images if total_images else 0
                    image_bar.progress(min(pct, 1.0))
                    image_log(f"{done_images}/{total_images} images generated — {message}")

                import pollinations_runner
                pollinations_token = get_secret("POLLINATIONS_TOKEN", pollinations_token_auto)

                active_manifest = st.session_state.get("manifest_path")
                if not active_manifest or not os.path.exists(active_manifest):
                    raise ValueError("Manifest path unresolved or non-existent.")

                force_fix_manifest_csv(active_manifest)

                images_dir, zip_path, failed_scenes = pollinations_runner.run_image_generation(
                    active_manifest, pollinations_token, progress_callback=on_image_progress
                )
                st.session_state["images_dir"] = images_dir
                image_bar.progress(1.0)
                if failed_scenes:
                    st.warning(f"{len(failed_scenes)} scenes failed and were skipped: {failed_scenes}. "
                               f"Re-run image generation to retry just those.")
                else:
                    st.success("All images generated.")

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
                    st.download_button("Download final_video.mp4", f, file_name="final_video.mp4", mime="video/mp4")

            except Exception as e:
                st.error(f"Pipeline stopped: {e}")

with tab_audio:
    default_script = st.session_state.get("script_text", "")
    script_for_audio = st.text_area("Script", default_script, height=300, key="audio_script")
    voice = st.selectbox("Voice", EDGE_TTS_VOICES)
    if st.button("Generate audio", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_audio_progress(done, total):
            progress_bar.progress(done / total if total else 0)
            status_text.markdown(f"**{done}/{total} audio segments** generated")

        try:
            path = generate_audio_file(script_for_audio, voice, progress_callback=on_audio_progress,
                                        work_dir=get_session_temp_dir())
            st.session_state["audio_path"] = path
            progress_bar.progress(1.0)
            st.success("Audio generated.")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
    if "audio_path" in st.session_state:
        st.audio(st.session_state["audio_path"])

with tab_align:
    default_script2 = st.session_state.get("script_text", "")
    script_for_align = st.text_area("Script", default_script2, height=300, key="align_script")
    uploaded_audio = st.file_uploader("Finished audio (optional)", type=["mp3", "wav"])
    if st.button("Generate scene manifest", type="primary"):
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
                    manifest_path, n_scenes = align_script_to_audio_file(script_for_align, audio_path,
                                                                          work_dir=get_session_temp_dir())
                    st.session_state["manifest_path"] = manifest_path
                    st.success(f"Aligned {n_scenes} scenes.")
                    with open(manifest_path, "rb") as f:
                        st.download_button("Download scene_manifest.csv", f, file_name="scene_manifest.csv")
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

    if st.button("Generate Images", type="primary"):
        if not active_manifest_path or not os.path.exists(active_manifest_path):
            st.error("Missing manifest file! Please upload a CSV or run alignment.")
        else:
            try:
                pollinations_token = get_secret("POLLINATIONS_TOKEN", pollinations_token_override)

                image_bar = st.progress(0.0)
                image_status = st.empty()
                image_log = make_progress_log(image_status)

                def on_image_progress(done_images, total_images, message):
                    pct = done_images / total_images if total_images else 0
                    image_bar.progress(min(pct, 1.0))
                    image_log(f"{done_images}/{total_images} images generated — {message}")

                import pollinations_runner

                force_fix_manifest_csv(active_manifest_path)

                images_dir, zip_path, failed_scenes = pollinations_runner.run_image_generation(
                    active_manifest_path, pollinations_token, progress_callback=on_image_progress
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
                        mime="application/zip"
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
            st.download_button("Download final_video.mp4", f, file_name="final_video.mp4", mime="video/mp4")

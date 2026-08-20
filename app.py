"""
app.py — Streamlit version, deployable free on Streamlit Community Cloud.

ZERO COST version: no Claude API calls anywhere in this app. Titles and
scripts are written in chat (free, part of normal Claude usage) and pasted
in here as a starting point. The app itself only ever calls genuinely free
services: edge-tts, local Whisper, and Kaggle's free GPU.

Required secrets (set in Streamlit Cloud: App settings -> Secrets):
  KAGGLE_USERNAME
  KAGGLE_KEY
(No Anthropic key needed — nothing in this app calls Claude.)
"""

import os
import re
import csv
import time
import asyncio
import tempfile
from difflib import SequenceMatcher

import streamlit as st

EDGE_TTS_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-AU-WilliamNeural",
]


def get_secret(key, override=""):
    if override.strip():
        return override.strip()
    try:
        return st.secrets.get(key, "")
    except Exception:
        return os.environ.get(key, "")


# ============================================================
# Core logic functions (unchanged from the Gradio version)
# ============================================================

def generate_audio_file(script_text, voice, progress_callback=None, scenes_per_chunk=7):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found — check the script has them.")

    # Group scenes into chunks (not one-per-scene) to preserve natural
    # prosody within each chunk — too-small chunks make speech sound choppy
    # at every boundary.
    chunks = [scenes[i:i + scenes_per_chunk] for i in range(0, len(scenes), scenes_per_chunk)]
    total_chunks = len(chunks)

    import edge_tts

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_text = " ".join(s["narration"] for s in chunk if s["narration"])
        if not chunk_text.strip():
            continue
        chunk_path = os.path.join(tempfile.gettempdir(), f"narration_chunk_{i}.mp3")

        async def _gen(text=chunk_text, path=chunk_path):
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(path)

        asyncio.run(_gen())
        chunk_paths.append(chunk_path)

        if progress_callback:
            progress_callback(i + 1, total_chunks)

    # Concatenate chunk audio files into one final file
    out_path = os.path.join(tempfile.gettempdir(), "narration.mp3")
    with open(out_path, "wb") as outfile:
        for cp in chunk_paths:
            with open(cp, "rb") as infile:
                outfile.write(infile.read())

    return out_path


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


def align_script_to_audio_file(script_text, audio_path):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found in the script.")

    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    whisper_words = []
    for seg in segments:
        for w in seg.words:
            whisper_words.append({"word": w.word.strip().lower(), "start": w.start, "end": w.end})

    def normalize(text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    plain_words = [w["word"] for w in whisper_words]
    cursor = 0
    for scene in scenes:
        target_words = normalize(scene["narration"])
        if not target_words:
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

    out_path = os.path.join(tempfile.gettempdir(), "scene_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "start_time", "end_time", "prompt"])
        for s in scenes:
            writer.writerow([s["scene_id"], seconds_to_hhmmss(s["start_time"]),
                              seconds_to_hhmmss(s["end_time"]), s["prompt"]])
    return out_path, len(scenes)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Faceless Channel Pipeline", layout="wide")
st.title("Faceless Channel Pipeline")
st.caption("Stick-figure style · unlimited characters · zero cost")

tab0, tab3, tab4, tab5 = st.tabs(
    ["1. Paste script", "2. Audio", "3. Align", "4. Images"])

with tab0:
    st.markdown(
        "Titles and scripts are written in chat (free) — see your Claude "
        "conversation, following the project rulebook. Paste the finished "
        "script here to start the pipeline."
    )
    pasted_script = st.text_area(
        "Script (must use [SCENE N: prompt] markers, as written in chat)",
        st.session_state.get("script_text", ""), height=400)
    col_a, col_b = st.columns(2)
    if col_a.button("Save script", type="primary"):
        if "[SCENE" not in pasted_script:
            st.warning("No [SCENE N: ...] markers found — check you pasted the full script.")
        else:
            st.session_state["script_text"] = pasted_script
            scene_count = len(re.findall(r"\[SCENE", pasted_script))
            narration_only = re.sub(r"\[SCENE.*?\]\n?", "", pasted_script, flags=re.DOTALL)
            word_count = len(re.findall(r"\S+", narration_only))
            st.success(f"Saved — {scene_count} scenes, ~{word_count} words, "
                       f"~{word_count/135:.1f} min estimated.")

    st.divider()
    st.markdown("**Or run the whole thing in one click** (audio → align → images), using the settings below.")
    voice_auto = st.selectbox("Voice", EDGE_TTS_VOICES, key="voice_auto")
    kaggle_user_auto = st.text_input("Kaggle username (leave blank if set in Secrets)", key="ku_auto")
    kaggle_key_auto = st.text_input("Kaggle API key (leave blank if set in Secrets)", type="password", key="kk_auto")
if col_b.button("Run full pipeline", type="primary"):
        if "[SCENE" not in pasted_script:
            st.warning("No [SCENE N: ...] markers found — check you pasted the full script.")
        else:
            st.session_state["script_text"] = pasted_script
            try:
                st.markdown("**Step 1/3 — audio**")
                audio_bar = st.progress(0.0)
                audio_status = st.empty()

                def on_audio_progress(done, total):
                    audio_bar.progress(done / total if total else 0)
                    audio_status.markdown(f"{done}/{total} audio segments")

                audio_path = generate_audio_file(pasted_script, voice_auto, progress_callback=on_audio_progress)
                st.session_state["audio_path"] = audio_path
                audio_bar.progress(1.0)
                st.success("Audio done.")

                with st.spinner("Step 2/3 — aligning to get timestamps..."):
                    manifest_path, n_scenes = align_script_to_audio_file(pasted_script, audio_path)
                    st.session_state["manifest_path"] = manifest_path
                st.success(f"Aligned {n_scenes} scenes.")

                st.markdown("**Step 3/3 — images**")
                image_bar = st.progress(0.0)
                image_status = st.empty()

                def on_image_progress(done_chunks, total_chunks, done_images, total_images, message):
                    pct = done_chunks / total_chunks if total_chunks else 0
                    image_bar.progress(pct)
                    image_status.markdown(f"{done_images}/{total_images} images — {message}")

                from kaggle_runner import run_image_generation_chunked
                kaggle_user = get_secret("KAGGLE_USERNAME", kaggle_user_auto)
                kaggle_key = get_secret("KAGGLE_KEY", kaggle_key_auto)
                
                # Check that manifest_path exists in session state before calling
                active_manifest = st.session_state.get("manifest_path", None)
                if not active_manifest:
                    raise ValueError("Manifest file was not created properly during alignment step.")

                zip_path = run_image_generation_chunked(active_manifest, kaggle_user, kaggle_key,
                                                          progress_callback=on_image_progress)
                image_bar.progress(1.0)
                st.success("All images generated.")
                with open(zip_path, "rb") as f:
                    st.download_button("Download scene_images_batch.zip", f, file_name="scene_images_batch.zip")
            except Exception as e:
                st.error(f"Pipeline stopped at an error: {e}")
                st.info("Progress up to the last completed step/chunk is saved — click 'Run full pipeline' again to resume rather than restart.")

with tab3:
    default_script = st.session_state.get("script_text", "")
    if default_script:
        st.info("Using the script saved in step 1 automatically.")
    script_for_audio = st.text_area("Script (auto-filled from step 1)",
                                     default_script, height=300, key="audio_script")
    voice = st.selectbox("Voice", EDGE_TTS_VOICES)
    if st.button("Generate audio", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_audio_progress(done, total):
            progress_bar.progress(done / total if total else 0)
            status_text.markdown(f"**{done}/{total} audio segments** generated")

        try:
            path = generate_audio_file(script_for_audio, voice, progress_callback=on_audio_progress)
            st.session_state["audio_path"] = path
            progress_bar.progress(1.0)
            st.success(f"Audio generated with voice {voice}.")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
    if "audio_path" in st.session_state:
        st.audio(st.session_state["audio_path"])

with tab4:
    default_script2 = st.session_state.get("script_text", "")
    script_for_align = st.text_area("Script (auto-filled from step 1)", default_script2, height=300, key="align_script")
    if "audio_path" in st.session_state:
        st.info("Using the audio generated in step 2 automatically. Upload a different file below only if you want to override it.")
    uploaded_audio = st.file_uploader("Finished audio (optional — auto-used from step 2 if you skip this)", type=["mp3", "wav"])
    if st.button("Generate scene manifest", type="primary"):
        if not uploaded_audio and "audio_path" not in st.session_state:
            st.warning("Upload an audio file, or generate one in the Audio tab first.")
        else:
            with st.spinner("Aligning (this can take a minute)..."):
                try:
                    if uploaded_audio:
                        audio_path = os.path.join(tempfile.gettempdir(), "uploaded_audio")
                        with open(audio_path, "wb") as f:
                            f.write(uploaded_audio.read())
                    else:
                        audio_path = st.session_state["audio_path"]
                    manifest_path, n_scenes = align_script_to_audio_file(script_for_align, audio_path)
                    st.session_state["manifest_path"] = manifest_path
                    st.success(f"Aligned {n_scenes} scenes.")
                    with open(manifest_path, "rb") as f:
                        st.download_button("Download scene_manifest.csv", f, file_name="scene_manifest.csv")
                except Exception as e:
                    st.error(f"Alignment failed: {e}")
with tab5:
    uploaded_manifest = st.file_uploader("scene_manifest.csv (optional — auto-used from step 3 if you skip this)", type=["csv"], key="tab5_uploader")

    # If user uploads a file directly in this tab, write it to temp storage and save path in session_state
    if uploaded_manifest is not None:
        temp_manifest_path = os.path.join(tempfile.gettempdir(), "uploaded_scene_manifest.csv")
        with open(temp_manifest_path, "wb") as f:
            f.write(uploaded_manifest.getvalue())
        st.session_state["manifest_path"] = temp_manifest_path

    active_manifest_path = st.session_state.get("manifest_path", None)

    if active_manifest_path and os.path.exists(active_manifest_path):
        st.info(f"Loaded manifest: {os.path.basename(active_manifest_path)}")

    kaggle_user_override = st.text_input("Kaggle username (leave blank if set in Secrets)", key="tab5_user")
    kaggle_key_override = st.text_input("Kaggle API key (leave blank if set in Secrets)", type="password", key="tab5_key")

    resume_info = None
    if active_manifest_path and os.path.exists(active_manifest_path):
        from kaggle_runner import get_resume_status
        try:
            resume_info = get_resume_status(active_manifest_path)
            if resume_info["has_progress"]:
                st.warning(
                    f"Found saved progress for this manifest: "
                    f"{resume_info['done_chunks']}/{resume_info['total_chunks']} chunks "
                    f"(~{resume_info['done_chunks'] * 25} of {resume_info['total_images']} images) already done. "
                    f"Clicking the button below will resume, not restart."
                )
        except Exception:
            pass

    button_label = "Resume image generation" if resume_info and resume_info["has_progress"] else "Generate images"

    if st.button(button_label, type="primary", key="btn_gen_images"):
        if not active_manifest_path or not os.path.exists(active_manifest_path):
            st.warning("No manifest found — run step 3 (Align) first, or upload a scene_manifest.csv here.")
        else:
            from kaggle_runner import run_image_generation_chunked
            kaggle_user = get_secret("KAGGLE_USERNAME", kaggle_user_override)
            kaggle_key = get_secret("KAGGLE_KEY", kaggle_key_override)

            if not kaggle_user or not kaggle_key:
                st.error("Missing Kaggle Credentials! Please enter your Kaggle Username and API Key.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def on_progress(done_chunks, total_chunks, done_images, total_images, message):
                    pct = done_chunks / total_chunks if total_chunks else 0
                    progress_bar.progress(pct)
                    status_text.markdown(f"**{done_images}/{total_images} images** "
                                          f"({done_chunks}/{total_chunks} chunks) — {message}")

                try:
                    zip_path = run_image_generation_chunked(
                        active_manifest_path, kaggle_user, kaggle_key,
                        progress_callback=on_progress)
                    progress_bar.progress(1.0)
                    st.success("All images generated.")
                    with open(zip_path, "rb") as f:
                        st.download_button("Download scene_images_batch.zip", f, file_name="scene_images_batch.zip")
                except Exception as e:
                    st.error(f"Stopped: {e}")
                    st.info("Progress up to the last completed chunk is saved. Click the button again to resume from there.")

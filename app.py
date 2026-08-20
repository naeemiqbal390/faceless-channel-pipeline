"""
app.py
Streamlit Web Application UI
"""

import os
import re
import csv
import json
import tempfile
import asyncio
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd


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


def generate_audio_file(script_text, voice, progress_callback=None, scenes_per_chunk=7):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found — check script input.")

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

    out_path = os.path.join(tempfile.gettempdir(), "narration.mp3")
    with open(out_path, "wb") as outfile:
        for cp in chunk_paths:
            with open(cp, "rb") as infile:
                outfile.write(infile.read())

    return out_path


def align_script_to_audio_file(script_text, audio_path):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found in script.")

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

    out_path = os.path.join(tempfile.gettempdir(), "scene_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "start_time", "end_time", "prompt"])
        for s in scenes:
            writer.writerow([s["scene_id"], seconds_to_hhmmss(s["start_time"]),
                              seconds_to_hhmmss(s["end_time"]), s["prompt"]])
    return out_path, len(scenes)


# Interface setup
st.set_page_config(page_title="Faceless Channel Pipeline", layout="wide")
st.title("Faceless Channel Pipeline")
st.caption("Stick-figure style · unlimited characters · zero cost")

if "manifest_path" not in st.session_state:
    st.session_state["manifest_path"] = None

tab0, tab3, tab4, tab5 = st.tabs(["1. Paste script", "2. Audio", "3. Align", "4. Images"])

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
    kaggle_user_auto = st.text_input("Kaggle username (or leave blank if set in Secrets)", key="ku_auto")
    kaggle_key_auto = st.text_input("Kaggle API key (or leave blank if set in Secrets)", type="password", key="kk_auto")

    if col_b.button("Run full pipeline", type="primary"):
        if "[SCENE" not in pasted_script:
            st.warning("No [SCENE N: ...] markers found.")
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

                with st.spinner("Step 2/3 — aligning..."):
                    manifest_path, n_scenes = align_script_to_audio_file(pasted_script, audio_path)
                    st.session_state["manifest_path"] = manifest_path

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

                active_manifest = st.session_state.get("manifest_path")
                if not active_manifest or not os.path.exists(active_manifest):
                    raise ValueError("Manifest path unresolved or non-existent.")

                force_fix_manifest_csv(active_manifest)

                zip_path = run_image_generation_chunked(active_manifest, kaggle_user, kaggle_key,
                                                         progress_callback=on_image_progress)
                image_bar.progress(1.0)
                st.success("All images generated.")
                with open(zip_path, "rb") as f:
                    st.download_button("Download scene_images_batch.zip", f, file_name="scene_images_batch.zip")
            except Exception as e:
                st.error(f"Pipeline stopped: {e}")

with tab3:
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
            path = generate_audio_file(script_for_audio, voice, progress_callback=on_audio_progress)
            st.session_state["audio_path"] = path
            progress_bar.progress(1.0)
            st.success("Audio generated.")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
    if "audio_path" in st.session_state:
        st.audio(st.session_state["audio_path"])

with tab4:
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
    st.markdown("### Generate Images on Kaggle")

    uploaded_manifest = st.file_uploader("scene_manifest.csv (optional)", type=["csv"], key="tab5_uploader")

    if uploaded_manifest is not None:
        temp_manifest_path = os.path.join(tempfile.gettempdir(), "scene_manifest.csv")
        with open(temp_manifest_path, "wb") as f:
            f.write(uploaded_manifest.getvalue())
        st.session_state["manifest_path"] = temp_manifest_path

    active_manifest_path = st.session_state.get("manifest_path")

    if active_manifest_path and os.path.exists(active_manifest_path):
        st.info(f"Loaded manifest ready: {os.path.basename(active_manifest_path)}")
        from kaggle_runner import get_resume_status
        try:
            resume_info = get_resume_status(active_manifest_path)
            if resume_info.get("has_progress"):
                st.success(f"Found saved progress: {resume_info['done_chunks']}/{resume_info['total_chunks']} chunks completed.")
        except Exception:
            pass
    else:
        st.warning("No active manifest found. Upload or generate a CSV manifest.")

    kaggle_user_override = st.text_input("Kaggle username (optional)", key="tab5_user")
    kaggle_key_override = st.text_input("Kaggle API key (optional)", type="password", key="tab5_key")

    if st.button("Generate Images on Kaggle", type="primary"):
        if not active_manifest_path or not os.path.exists(active_manifest_path):
            st.error("Missing manifest file! Please upload a CSV or run alignment.")
        else:
            try:
                kaggle_user = get_secret("KAGGLE_USERNAME", kaggle_user_override)
                kaggle_key = get_secret("KAGGLE_KEY", kaggle_key_override)

                if not kaggle_user or not kaggle_key:
                    st.error("Kaggle credentials missing.")
                else:
                    image_bar = st.progress(0.0)
                    image_status = st.empty()

                    def on_image_progress(done_chunks, total_chunks, done_images, total_images, message):
                        pct = done_chunks / total_chunks if total_chunks else 0
                        image_bar.progress(pct)
                        image_status.markdown(f"**{done_images}/{total_images} images** — {message}")

                    from kaggle_runner import run_image_generation_chunked

                    force_fix_manifest_csv(active_manifest_path)

                    zip_path = run_image_generation_chunked(
                        active_manifest_path,
                        kaggle_user,
                        kaggle_key,
                        progress_callback=on_image_progress
                    )

                    image_bar.progress(1.0)
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

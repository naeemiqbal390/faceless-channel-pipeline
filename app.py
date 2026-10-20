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
import time
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


def _get_project_status_uncached(project_id):
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


@st.cache_data(ttl=45, show_spinner=False)
def get_project_status(project_id):
    """Cached for 45 s: this B2 listing used to run on EVERY click, because Streamlit re-runs the whole
    page (all tabs) each time. Cleared whenever progress changes (Resume, Clear, image generation)."""
    return _get_project_status_uncached(project_id)


def resume_project_from_b2(project_id):
    """Downloads every saved artifact for this project into the current
    session's local workspace and restores session_state to match.
    Returns a summary dict that the UI shows, so a restore is never silent.
    A saved video is only restored if EVERY scene image exists; a video built
    from an incomplete image set is stale, so it is discarded."""
    import b2_storage
    session_dir = get_session_temp_dir()
    prefix = f"projects/{project_id}/"
    summary = {"audio": False, "manifest": False, "total": 0, "have": 0,
               "download_failed": 0, "video": False, "video_discarded": False}

    audio_key = prefix + "narration.mp3"
    if b2_storage.key_exists(audio_key):
        local_audio = os.path.join(session_dir, "narration.mp3")
        if b2_storage.download_file(audio_key, local_audio):
            st.session_state["audio_path"] = local_audio
            summary["audio"] = True

    manifest_key = prefix + "scene_manifest.csv"
    local_manifest = None
    if b2_storage.key_exists(manifest_key):
        local_manifest = os.path.join(session_dir, "scene_manifest.csv")
        if b2_storage.download_file(manifest_key, local_manifest):
            st.session_state["manifest_path"] = local_manifest
            summary["manifest"] = True
        else:
            local_manifest = None

    if local_manifest:
        import cloudflare_runner
        images_dir = cloudflare_runner.get_images_dir(local_manifest)
        image_keys = [k for k in b2_storage.list_keys(prefix + "scene_images/") if k.endswith(".png")]
        for k in image_keys:
            dest = os.path.join(images_dir, os.path.basename(k))
            if os.path.exists(dest):
                continue
            ok = b2_storage.download_file(k, dest) or b2_storage.download_file(k, dest)
            if not ok:
                summary["download_failed"] += 1
        st.session_state["images_dir"] = images_dir
        status = cloudflare_runner.get_resume_status(local_manifest)
        summary["total"] = status["total_images"]
        summary["have"] = status["done_images"]

    video_key = prefix + "final_video.mp4"
    if b2_storage.key_exists(video_key):
        complete = summary["total"] > 0 and summary["have"] >= summary["total"]
        if complete:
            local_video = os.path.join(session_dir, "final_video.mp4")
            if b2_storage.download_file(video_key, local_video):
                st.session_state["video_path"] = local_video
                summary["video"] = True
        else:
            # stale video made before all images existed — remove it
            b2_storage.delete_prefix(video_key)
            st.session_state.pop("video_path", None)
            summary["video_discarded"] = True

    st.session_state["project_id"] = project_id
    return summary


def sync_images_to_b2(images_dir):
    """Uploads any local scene image that B2 doesn't have yet (with retries),
    so a restart can never lose images that were generated. Returns count uploaded."""
    script_text = st.session_state.get("script_text", "")
    if not script_text or not images_dir or not os.path.isdir(images_dir):
        return 0
    try:
        import b2_storage
        if not b2_storage.is_configured():
            return 0
        prefix = f"projects/{compute_project_id(script_text)}/scene_images/"
        have = {os.path.basename(k) for k in b2_storage.list_keys(prefix)}
        uploaded = 0
        for f in sorted(os.listdir(images_dir)):
            if f.lower().endswith(".png") and f not in have:
                for _ in range(3):
                    if b2_storage.upload_file(os.path.join(images_dir, f), prefix + f):
                        uploaded += 1
                        break
        return uploaded
    except Exception:
        return 0


def invalidate_saved_video(manifest_path=None):
    """Removes any previously built video (local + B2). Called whenever images
    are still missing, because such a video is incomplete by definition."""
    script_text = st.session_state.get("script_text", "")
    if not script_text:
        return
    try:
        if manifest_path:
            import cloudflare_runner
            st_ = cloudflare_runner.get_resume_status(manifest_path)
            if st_["total_images"] and st_["done_images"] >= st_["total_images"]:
                return  # nothing missing — a saved video is still valid
        st.session_state.pop("video_path", None)
        import b2_storage
        if b2_storage.is_configured():
            b2_storage.delete_prefix(f"projects/{compute_project_id(script_text)}/final_video.mp4")
    except Exception:
        pass


def make_background_uploader(workers=3):
    """Uploads each finished image to B2 in background threads, so image generation is no longer
    paused while every picture uploads. Returns (submit, finish); call finish() when generation ends.
    Anything that fails here is caught later by sync_images_to_b2()."""
    from concurrent.futures import ThreadPoolExecutor
    import b2_storage
    script_text = st.session_state.get("script_text", "")
    enabled = bool(script_text) and b2_storage.is_configured()
    pid = compute_project_id(script_text) if enabled else None
    pool = ThreadPoolExecutor(max_workers=workers) if enabled else None

    def submit(local_path, relative_name):
        if not enabled:
            return

        def job():
            for _ in range(3):
                try:
                    if b2_storage.upload_file(local_path, f"projects/{pid}/{relative_name}"):
                        return
                except Exception:
                    pass
                time.sleep(1)
        pool.submit(job)

    def finish():
        if pool:
            pool.shutdown(wait=True)

    return submit, finish


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
        for _ in range(3):
            if b2_storage.upload_file(local_path, f"projects/{project_id}/{relative_name}"):
                break
    except Exception:
        pass


def save_script_to_b2(script_text):
    """
    Saves the script itself plus a 'latest project' pointer to B2, so that
    after a reboot the app can reopen the last project WITHOUT the script
    being pasted again. Best-effort: never raises.
    """
    try:
        import b2_storage
        if not script_text or not b2_storage.is_configured():
            return
        project_id = compute_project_id(script_text)
        tmp_dir = get_session_temp_dir()
        script_file = os.path.join(tmp_dir, "script.txt")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(script_text)
        b2_storage.upload_file(script_file, f"projects/{project_id}/script.txt")
        pointer_file = os.path.join(tmp_dir, "_latest.json")
        with open(pointer_file, "w", encoding="utf-8") as f:
            json.dump({"project_id": project_id, "updated": time.time()}, f)
        b2_storage.upload_file(pointer_file, "projects/_latest.json")
    except Exception:
        pass


def autoload_last_project():
    """
    Runs once per browser session. If this session has no script yet and B2
    holds a 'latest project' pointer, restore that project (script, audio,
    manifest, images, video) so the app opens exactly where it was left.
    """
    if st.session_state.get("autoload_done"):
        return
    st.session_state["autoload_done"] = True
    if st.session_state.get("script_text"):
        return
    try:
        import b2_storage
        if not b2_storage.is_configured():
            return
        pointer_local = os.path.join(get_session_temp_dir(), "_latest.json")
        if not b2_storage.download_file("projects/_latest.json", pointer_local):
            return
        with open(pointer_local, "r", encoding="utf-8") as f:
            project_id = json.load(f).get("project_id")
        if not project_id:
            return
        script_local = os.path.join(get_session_temp_dir(), "script_restored.txt")
        if not b2_storage.download_file(f"projects/{project_id}/script.txt", script_local):
            return
        with open(script_local, "r", encoding="utf-8") as f:
            st.session_state["script_text"] = f.read()
        with st.spinner("Restoring your last project from B2..."):
            st.session_state["resume_summary"] = resume_project_from_b2(project_id)
        st.session_state["autoloaded_project"] = project_id
    except Exception as e:
        print(f"[autoload] could not restore last project: {e}", flush=True)


def build_srt(script_text, manifest_path):
    """
    Builds an .srt caption file from the script's narration and the manifest's
    scene start times. Each scene's text is split into short caption lines;
    the scene's time window (its start to the next scene's start) is shared
    out in proportion to text length. Word-level timing isn't stored, so
    caption breaks are approximate to within a second or so.
    """
    scenes = {sc["scene_id"]: sc["narration"] for sc in parse_script_scenes(script_text)}
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip().str.lower()

    def to_sec(v):
        try:
            h, m, sec = str(v).split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        except Exception:
            return None

    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        try:
            sid = int(row["scene_id"])
        except Exception:
            sid = i + 1
        rows.append((sid, to_sec(row.get("start_time")), to_sec(row.get("end_time"))))

    def fmt(t):
        t = max(0.0, t)
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    def split_caption_lines(text, max_chars=84):
        words, chunks, cur = text.split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                chunks.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            chunks.append(cur)
        return chunks

    def two_lines(chunk, width=42):
        if len(chunk) <= width:
            return chunk
        mid = chunk.rfind(" ", 0, width + 1)
        return chunk if mid <= 0 else chunk[:mid] + "\n" + chunk[mid + 1:]

    out, n = [], 1
    for idx, (sid, start, end) in enumerate(rows):
        text = (scenes.get(sid) or "").strip()
        if not text or start is None:
            continue
        nxt = rows[idx + 1][1] if idx + 1 < len(rows) and rows[idx + 1][1] is not None else None
        window_end = nxt if nxt is not None else (end if end is not None else start + 4.0)
        if window_end <= start:
            window_end = start + 2.0
        chunks = split_caption_lines(text)
        total_chars = sum(len(c) for c in chunks) or 1
        t = start
        for c in chunks:
            dur = (window_end - start) * len(c) / total_chars
            out.append(f"{n}\n{fmt(t)} --> {fmt(t + dur)}\n{two_lines(c)}\n")
            n += 1
            t += dur
    return "\n".join(out)


@st.cache_data(show_spinner=False, max_entries=4)
def build_srt_cached(script_text, manifest_path, manifest_mtime):
    """build_srt re-reads the manifest and re-parses the script; cache it so clicks elsewhere don't repeat that."""
    return build_srt(script_text, manifest_path)


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


def get_cloudflare_accounts(manual_id="", manual_token=""):
    """
    Ordered list of Cloudflare accounts: [(label, account_id, token), ...].
    Order is fixed: account 1 (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN), then
    CLOUDFLARE_ACCOUNT_ID_2 / CLOUDFLARE_API_TOKEN_2, ... up to _6.
    The app uses account 1 until its daily quota is gone, then moves to 2, etc.
    """
    accounts = []
    first_id = get_secret(["CLOUDFLARE_ACCOUNT_ID"], manual_id)
    first_token = get_secret(["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY"], manual_token)
    if first_id and first_token:
        accounts.append(("Account 1", first_id, first_token))
    for n in range(2, 7):
        a_id = get_secret([f"CLOUDFLARE_ACCOUNT_ID_{n}"])
        a_tok = get_secret([f"CLOUDFLARE_API_TOKEN_{n}"])
        if a_id and a_tok:
            accounts.append((f"Account {n}", a_id, a_tok))
    return accounts


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


PIPER_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx"


def _generate_audio_piper(chunks, work_dir, progress_callback, total_chunks, edge_error):
    import subprocess, sys, requests
    model_path = os.path.join(tempfile.gettempdir(), "piper_en_US-ryan-high.onnx")
    for url, path in ((PIPER_MODEL_URL, model_path), (PIPER_MODEL_URL + ".json", model_path + ".json")):
        if not os.path.exists(path):
            print(f"[audio] downloading backup voice file: {url}", flush=True)
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

    wav_paths = []
    for i, chunk in enumerate(chunks):
        text = " ".join(s["narration"] for s in chunk if s["narration"]).strip()
        if not text:
            continue
        txt_path = os.path.join(work_dir, f"piper_chunk_{i}.txt")
        wav_path = os.path.join(work_dir, f"piper_chunk_{i}.wav")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        proc = subprocess.run(
            [sys.executable, "-m", "piper", "-m", model_path, "-i", txt_path, "-f", wav_path],
            capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(wav_path):
            raise RuntimeError(
                f"Edge-TTS failed ({edge_error}) and the backup voice failed too: {proc.stderr[-400:]}")
        wav_paths.append(wav_path)
        if progress_callback:
            progress_callback(i + 1, total_chunks)

    list_path = os.path.join(work_dir, "piper_concat.txt")
    with open(list_path, "w") as f:
        for wp in wav_paths:
            f.write(f"file '{wp}'\n")
    out_path = os.path.join(work_dir, "narration.mp3")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                    "-c:a", "libmp3lame", "-b:a", "192k", out_path],
                   check=True, capture_output=True)
    return out_path


def generate_audio_file(script_text, voice, progress_callback=None, scenes_per_chunk=7, work_dir=None):
    scenes = parse_script_scenes(script_text)
    if not scenes:
        raise ValueError("No [SCENE N: ...] markers found — check script input.")

    work_dir = work_dir or tempfile.gettempdir()
    chunks = [scenes[i:i + scenes_per_chunk] for i in range(0, len(scenes), scenes_per_chunk)]
    total_chunks = len(chunks)

    import edge_tts

    chunk_paths = []
    edge_failed = None
    for i, chunk in enumerate(chunks):
        chunk_text = " ".join(s["narration"] for s in chunk if s["narration"])
        if not chunk_text.strip():
            continue
        chunk_path = os.path.join(work_dir, f"narration_chunk_{i}.mp3")

        async def _gen(text=chunk_text, path=chunk_path):
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(path)

        # Microsoft's endpoint sometimes answers 403 transiently (or blocks a
        # cloud IP for a while) — retry with growing waits before giving up.
        last_err = None
        for wait in (0, 2, 5):
            if wait:
                time.sleep(wait)
            try:
                asyncio.run(_gen())
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[audio] Edge-TTS chunk {i} failed: {str(e)[:150]}", flush=True)
        if last_err is not None:
            edge_failed = last_err
            print("[audio] Edge-TTS is blocked — switching to backup voice (Piper).", flush=True)
            break
        chunk_paths.append(chunk_path)

        if progress_callback:
            progress_callback(i + 1, total_chunks)

    if edge_failed is not None:
        # Edge-TTS is refusing this server. Fall back to Piper (offline
        # neural voice) for the WHOLE narration so the voice stays consistent.
        return _generate_audio_piper(chunks, work_dir, progress_callback, total_chunks, edge_failed)

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

    shown = {"t": 0.0}

    def _log(message):
        lines.append(message)
        print(message, flush=True)
        # fewer UI pushes = less load on the free host; the full log still goes to the console
        now = time.time()
        if now - shown["t"] < 1.0 and "quota" not in message.lower() and "Switch" not in message:
            return
        shown["t"] = now
        try:
            container.code("\n".join(lines[-max_lines_shown:]), language=None, wrap_lines=True)
        except TypeError:
            container.code("\n".join(lines[-max_lines_shown:]), language=None)

    return _log


def show_video_preview(video_path, key):
    """The video file can be hundreds of MB, and Streamlit re-reads whatever is shown on EVERY click
    (this page showed it in two tabs). So the player and download button load only when asked."""
    try:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
    except OSError:
        return
    if st.checkbox(f"Show video player and download ({size_mb:.0f} MB; leave off to keep the app fast)",
                   key=f"show_video_{key}", value=False):
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button("Download final_video.mp4", f, file_name="final_video.mp4",
                               mime="video/mp4", key=f"dl_video_{key}")


def make_image_gallery(container, images_dir, cols=5, max_shown=15):
    """
    Renders the most recently generated images as a thumbnail grid,
    refreshed on demand — lets you actually see quality along the way
    instead of waiting for the whole batch to finish. Reads the images
    directory fresh each call rather than tracking state, since
    cloudflare_runner writes deterministic scene_NNN.png filenames.
    """
    import base64
    import io
    from PIL import Image

    cache = st.session_state.setdefault("_thumb_cache", {})

    def _thumb_b64(path):
        # Keyed by path + mtime + size so a changed file can never show a stale thumbnail.
        try:
            st_info = os.stat(path)
            key = (path, st_info.st_mtime_ns, st_info.st_size)
        except OSError:
            return None
        if key in cache:
            return cache[key]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((260, 260))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=70)
            data = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None
        if len(cache) > 400:
            cache.clear()
        cache[key] = data
        return data

    last = {"t": 0.0}

    def _refresh(force=False):
        # Redrawing the gallery on every log line floods the browser connection
        # and burns CPU on the free host, so redraw at most every few seconds.
        now = time.time()
        if not force and last["t"] and now - last["t"] < 8:
            return
        last["t"] = now
        try:
            entries = []
            for f in os.listdir(images_dir):
                if f.lower().endswith(".png"):
                    p = os.path.join(images_dir, f)
                    entries.append((os.path.getmtime(p), f))
            entries.sort()
        except Exception:
            entries = []
        # most recently generated images, shown newest first
        recent = [f for _, f in entries[-max_shown:]][::-1]
        if not recent:
            container.caption("Images will appear here as they're generated.")
            return
        cells = []
        for fname in recent:
            b64 = _thumb_b64(os.path.join(images_dir, fname))
            if not b64:
                continue
            cells.append(
                '<div style="text-align:center;font-size:12px">'
                f'<img src="data:image/jpeg;base64,{b64}" '
                'style="width:100%;border-radius:6px;display:block"/>'
                f'{fname.replace(".png", "")}</div>'
            )
        html = (f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:8px">'
                + "".join(cells) + "</div>")
        container.markdown(html, unsafe_allow_html=True)

    return _refresh


# Interface setup
st.set_page_config(page_title="Faceless Channel Pipeline", layout="wide")
st.title("Faceless Channel Pipeline")
st.caption("Stick-figure style · unlimited characters · zero cost")

if "manifest_path" not in st.session_state:
    st.session_state["manifest_path"] = None

autoload_last_project()
if st.session_state.get("autoloaded_project"):
    st.success("Reopened your last project exactly where you left it. Use the tabs below to continue.")

tab0, tab_audio, tab_align, tab_images, tab_video, tab_thumb = st.tabs(
    ["1. Paste script", "2. Audio", "3. Align", "4. Images", "5. Video", "6. Thumbnail"]
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
            save_script_to_b2(pasted_script)
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
                    "video saved (checked against images on Resume)" if project_status["has_video"] else "video ✗",
                ]
                st.info(f"Saved progress found for this script: {', '.join(status_bits)}")
                resume_col, clear_col = st.columns(2)
                if resume_col.button("Resume — load saved progress", type="primary"):
                    with st.spinner("Downloading saved progress from B2..."):
                        st.session_state["resume_summary"] = resume_project_from_b2(check_project_id)
                    get_project_status.clear()
                    st.rerun()
                if clear_col.button("Clear saved progress — back to stage 1"):
                    b2_storage.delete_prefix(f"projects/{check_project_id}/")
                    get_project_status.clear()
                    try:
                        b2_storage.delete_prefix("projects/_latest.json")
                    except Exception:
                        pass
                    st.session_state["autoload_done"] = True
                    for key in ["script_text", "audio_path", "manifest_path", "images_dir",
                                "video_path", "project_id", "resume_summary"]:
                        st.session_state.pop(key, None)
                    st.success("Cleared. Starting over.")
                    st.rerun()
        else:
            st.caption("Add B2_KEY_ID / B2_APPLICATION_KEY / B2_BUCKET_NAME / B2_ENDPOINT_URL to Secrets to "
                       "enable save/resume across sessions and restarts.")

    # Shown right here, in the same tab, whenever anything is loaded —
    # right after Resume, or after any pipeline step completes — so there's
    # no need to go hunting through other tabs to confirm something worked.
    rs = st.session_state.get("resume_summary")
    if rs:
        if rs["total"]:
            missing = rs["total"] - rs["have"]
            msg = (f"Restored: audio {'✓' if rs['audio'] else '✗'}, manifest {'✓' if rs['manifest'] else '✗'}, "
                   f"images {rs['have']}/{rs['total']}" + (f" ({missing} still to generate)" if missing else " (complete)"))
            (st.success if not missing else st.warning)(msg)
        else:
            st.warning("Restored what was saved, but no scene manifest was found yet.")
        if rs["download_failed"]:
            st.error(f"{rs['download_failed']} saved image(s) could not be downloaded from B2 — they will be regenerated.")
        if rs["video_discarded"]:
            st.info("An older video that was built before all images existed was removed. "
                    "A fresh video is built only when every image is done.")
        if rs["video"]:
            st.success("Complete video restored.")

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
            all_pngs = sorted(f for f in os.listdir(images_d) if f.lower().endswith(".png"))
            with st.expander(f"{len(all_pngs)} image(s) available — click to review", expanded=False):
                # Building this grid sends ~0.5 MB of HTML to the browser on every click, even while the
                # expander is closed, so it only loads when ticked.
                if st.checkbox("Load thumbnails", key="show_status_gallery", value=False):
                    status_gallery_placeholder = st.empty()
                    make_image_gallery(status_gallery_placeholder, images_d, max_shown=40)()
        video_p = st.session_state.get("video_path")
        if video_p and os.path.exists(video_p):
            show_video_preview(video_p, "tab0_status")

    st.divider()
    st.checkbox("Show live image gallery while generating (turn off if the app gets throttled)",
                value=True, key="show_live_gallery")
    voice_auto = st.selectbox("Voice", EDGE_TTS_VOICES, key="voice_auto")

    import cloudflare_runner
    style_options = list(cloudflare_runner.STYLE_PRESETS.keys())
    style_auto = st.selectbox("Image style", style_options, key="style_auto")
    st.session_state["image_style"] = style_auto

    cloudflare_account_auto = st.text_input(
        "Cloudflare Account ID (optional — or set CLOUDFLARE_ACCOUNT_ID in Secrets)",
        key="cfa_auto",
    )
    cloudflare_token_auto = st.text_input(
        "Cloudflare API token (optional — or set CLOUDFLARE_API_TOKEN in Secrets)",
        type="password", key="cft_auto",
    )

    if col_b.button("Run full pipeline", type="primary"):
        if "[SCENE" not in pasted_script:
            st.warning("No [SCENE N: ...] markers found.")
        else:
            st.session_state["script_text"] = pasted_script
            save_script_to_b2(pasted_script)
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
                import cloudflare_runner
                cloudflare_accounts = get_cloudflare_accounts(cloudflare_account_auto, cloudflare_token_auto)
                if not cloudflare_accounts:
                    raise ValueError(
                        "Cloudflare credentials missing — add CLOUDFLARE_ACCOUNT_ID and "
                        "CLOUDFLARE_API_TOKEN to Secrets (or enter them in the fields above)."
                    )
                st.caption("Image account order: " + " → ".join(a[0] for a in cloudflare_accounts)
                           + ". Each is used until its daily quota is gone, then the next takes over.")

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
                images_dir_preview = cloudflare_runner.get_images_dir(active_manifest)

                log_col, gallery_col = st.columns([2, 3])
                with log_col:
                    image_bar = st.progress(0.0)
                    image_status = st.empty()
                image_log = make_progress_log(image_status)
                with gallery_col:
                    st.caption("Generated so far (most recent):")
                    gallery_placeholder = st.empty()
                refresh_gallery = make_image_gallery(gallery_placeholder, images_dir_preview)
                if not st.session_state.get("show_live_gallery", True):
                    _g = refresh_gallery
                    refresh_gallery = lambda force=False: (_g(force=True) if force else None)
                refresh_gallery()

                def on_image_progress(done_images, total_images, message):
                    pct = done_images / total_images if total_images else 0
                    image_bar.progress(min(pct, 1.0))
                    image_log(f"{done_images}/{total_images} images generated — {message}")
                    refresh_gallery()

                _bg_submit, _bg_finish = make_background_uploader()

                def on_image_saved(scene_id, local_path):
                    _bg_submit(local_path, f"scene_images/{os.path.basename(local_path)}")

                invalidate_saved_video(active_manifest)
                images_dir, zip_path, failed_scenes = cloudflare_runner.run_image_generation(
                    active_manifest, progress_callback=on_image_progress,
                    style=st.session_state.get("image_style", cloudflare_runner.DEFAULT_STYLE),
                    on_image_saved=on_image_saved,
                    accounts=cloudflare_accounts,
                )
                st.session_state["images_dir"] = images_dir
                refresh_gallery(force=True)
                _bg_finish()
                sync_images_to_b2(images_dir)
                get_project_status.clear()
                _status = cloudflare_runner.get_resume_status(active_manifest)
                _missing = _status["total_images"] - _status["done_images"]
                if _missing > 0 or failed_scenes:
                    image_bar.progress(min(_status["done_images"] / max(_status["total_images"], 1), 1.0))
                    st.error(
                        f"Stopped before the video: {_status['done_images']}/{_status['total_images']} images are done, "
                        f"{_missing} still missing (likely today's free Cloudflare quota ran out). "
                        f"Nothing is lost — everything is saved. Come back after 05:00 Pakistan time "
                        f"(00:00 UTC, when the quota resets), reopen the app and press Resume. "
                        f"The video will only be built once ALL images exist.")
                    with open(zip_path, "rb") as f:
                        st.download_button("Download images so far (zip)", f, file_name="scene_images_batch.zip",
                                            mime="application/zip", key="dl_images_pipeline_partial")
                    st.stop()
                image_bar.progress(1.0)
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
    st.markdown("### Generate Images (Cloudflare Workers AI)")
    st.caption("Full-step SDXL diffusion (Leonardo Phoenix tried first where your account has it, "
               "flux-schnell as a fast last resort). Free tier is 10,000 Neurons/day, no card, "
               "resetting at 00:00 UTC — roughly 40-50 full-quality images/day. Once a day's "
               "quota runs out, remaining scenes stay parked; come back after the reset and "
               "click Generate again (or Resume) to continue exactly where it stopped.")

    uploaded_manifest = st.file_uploader("scene_manifest.csv (optional)", type=["csv"], key="tab_images_uploader")

    if uploaded_manifest is not None:
        temp_manifest_path = os.path.join(get_session_temp_dir(), "scene_manifest.csv")
        with open(temp_manifest_path, "wb") as f:
            f.write(uploaded_manifest.getvalue())
        st.session_state["manifest_path"] = temp_manifest_path

    active_manifest_path = st.session_state.get("manifest_path")

    if active_manifest_path and os.path.exists(active_manifest_path):
        st.info(f"Loaded manifest ready: {os.path.basename(active_manifest_path)}")
        import cloudflare_runner
        try:
            resume_info = cloudflare_runner.get_resume_status(active_manifest_path)
            if resume_info.get("has_progress"):
                st.success(f"Found saved progress: {resume_info['done_images']}/{resume_info['total_images']} images already done.")
        except Exception:
            pass
    else:
        st.warning("No active manifest found. Upload or generate a CSV manifest.")

    cloudflare_account_override = st.text_input(
        "Cloudflare Account ID (optional — or set CLOUDFLARE_ACCOUNT_ID in Secrets)",
        key="tab_images_account",
    )
    cloudflare_token_override = st.text_input(
        "Cloudflare API token (optional — or set CLOUDFLARE_API_TOKEN in Secrets)",
        type="password", key="tab_images_key",
    )

    import cloudflare_runner
    style_options = list(cloudflare_runner.STYLE_PRESETS.keys())
    default_style = st.session_state.get("image_style", cloudflare_runner.DEFAULT_STYLE)
    default_index = style_options.index(default_style) if default_style in style_options else 0
    style_override = st.selectbox("Image style", style_options, index=default_index, key="tab_images_style")
    st.session_state["image_style"] = style_override

    if st.button("Generate Images", type="primary"):
        if not active_manifest_path or not os.path.exists(active_manifest_path):
            st.error("Missing manifest file! Please upload a CSV or run alignment.")
        else:
            try:
                cloudflare_accounts = get_cloudflare_accounts(cloudflare_account_override, cloudflare_token_override)
                if not cloudflare_accounts:
                    raise ValueError(
                        "Cloudflare credentials missing — add CLOUDFLARE_ACCOUNT_ID and "
                        "CLOUDFLARE_API_TOKEN to Secrets (or enter them in the fields above)."
                    )
                st.caption("Image account order: " + " → ".join(a[0] for a in cloudflare_accounts)
                           + ". Each is used until its daily quota is gone, then the next takes over.")

                import cloudflare_runner
                force_fix_manifest_csv(active_manifest_path)
                images_dir_preview = cloudflare_runner.get_images_dir(active_manifest_path)

                log_col, gallery_col = st.columns([2, 3])
                with log_col:
                    image_bar = st.progress(0.0)
                    image_status = st.empty()
                image_log = make_progress_log(image_status)
                with gallery_col:
                    st.caption("Generated so far (most recent):")
                    gallery_placeholder = st.empty()
                refresh_gallery = make_image_gallery(gallery_placeholder, images_dir_preview)
                if not st.session_state.get("show_live_gallery", True):
                    _g = refresh_gallery
                    refresh_gallery = lambda force=False: (_g(force=True) if force else None)
                refresh_gallery()

                def on_image_progress(done_images, total_images, message):
                    pct = done_images / total_images if total_images else 0
                    image_bar.progress(min(pct, 1.0))
                    image_log(f"{done_images}/{total_images} images generated — {message}")
                    refresh_gallery()

                _bg_submit, _bg_finish = make_background_uploader()

                def on_image_saved(scene_id, local_path):
                    _bg_submit(local_path, f"scene_images/{os.path.basename(local_path)}")

                invalidate_saved_video(active_manifest_path)
                images_dir, zip_path, failed_scenes = cloudflare_runner.run_image_generation(
                    active_manifest_path, progress_callback=on_image_progress,
                    style=style_override,
                    on_image_saved=on_image_saved,
                    accounts=cloudflare_accounts,
                )
                st.session_state["images_dir"] = images_dir

                refresh_gallery(force=True)
                _bg_finish()
                sync_images_to_b2(images_dir)
                get_project_status.clear()
                _status = cloudflare_runner.get_resume_status(active_manifest_path)
                _missing = _status["total_images"] - _status["done_images"]
                if _missing > 0 or failed_scenes:
                    image_bar.progress(min(_status["done_images"] / max(_status["total_images"], 1), 1.0))
                    st.error(
                        f"{_status['done_images']}/{_status['total_images']} images done, {_missing} missing "
                        f"(likely today's free quota ran out). Everything is saved. After 05:00 Pakistan time "
                        f"(00:00 UTC) click Generate Images again — only the missing ones are made. "
                        f"The video cannot be built until all images exist.")
                else:
                    image_bar.progress(1.0)
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
        import cloudflare_runner
        images_dir_for_video = cloudflare_runner.get_images_dir(active_manifest_for_video)
        resume_info = cloudflare_runner.get_resume_status(active_manifest_for_video)
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
        elif resume_info["done_images"] < resume_info["total_images"]:
            st.error(f"Cannot build the video yet: {resume_info['total_images'] - resume_info['done_images']} "
                     f"of {resume_info['total_images']} images are still missing. Finish them in the Images tab "
                     f"(after 05:00 Pakistan time when the quota resets), then assemble.")
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
        show_video_preview(st.session_state["video_path"], "video_tab")

    st.markdown("#### Captions (.srt)")
    st.caption("YouTube also makes automatic captions on its own after upload (usually within minutes to a few "
               "hours), but they can misspell names and terms. This file is built from your own script, so the "
               "wording is exact. In YouTube Studio: Subtitles > Add language > Upload file > With timing.")
    _script_for_srt = st.session_state.get("script_text", "")
    _manifest_for_srt = st.session_state.get("manifest_path")
    if _script_for_srt and _manifest_for_srt and os.path.exists(_manifest_for_srt):
        try:
            srt_text = build_srt_cached(_script_for_srt, _manifest_for_srt, os.path.getmtime(_manifest_for_srt))
            st.download_button("Download captions.srt", srt_text.encode("utf-8"), file_name="captions.srt",
                               mime="application/x-subrip", key="dl_srt_tab")
        except Exception as e:
            st.warning(f"Could not build captions: {e}")
    else:
        st.caption("Captions appear here once the script and alignment manifest are loaded.")

    # ---- YouTube chapters ----
    st.markdown("#### Chapters (timestamps for the description)")
    st.caption("Chapters show as sections on YouTube's progress bar. Press the button for a proposal built from your "
               "real scene timings at the story's turning points, edit the titles (or times) freely, then paste "
               "the list at the top of your video description. Keep titles teasers so they don't spoil twists.")
    if _script_for_srt and _manifest_for_srt and os.path.exists(_manifest_for_srt):
        if st.button("Suggest chapters", key="chapters_suggest"):
            try:
                import chapters as _ch
                import thumbnail_studio as _ts
                _df = pd.read_csv(_manifest_for_srt, encoding="utf-8-sig")
                _df.columns = _df.columns.astype(str).str.strip().str.lower()
                _starts, _total = _ch.scene_timings(_df)
                _narr = {sc["scene_id"]: sc["narration"] for sc in parse_script_scenes(_script_for_srt)}
                _props = _ch.propose_chapters(_starts, _total, _narr)
                _titles = [p[2] for p in _props]
                _note = ""
                _gkey = get_secret(["GEMINI_API_KEY"])
                if _gkey and _props:
                    _ai, _err = _ch.ai_titles(_ch.section_excerpts(_props, _starts, _narr),
                                              st.session_state.get("thumb_title", ""), _gkey,
                                              _ts._gemini_json, get_secret(["GEMINI_MODEL"]))
                    if _ai:
                        _titles = _ai
                    else:
                        _note = f"Used simple default titles ({_err})."
                _long = _total >= 3600
                st.session_state["chapters_text"] = "\n".join(
                    f"{_ch.format_timestamp(p[0], _long)} {_titles[i]}" for i, p in enumerate(_props))
                st.session_state["chapters_total"] = _total
                st.session_state["chapters_note"] = _note
            except Exception as e:
                st.warning(f"Could not build chapters: {e}")
        if st.session_state.get("chapters_note"):
            st.caption(st.session_state["chapters_note"])
        if st.session_state.get("chapters_text"):
            import chapters as _ch
            edited = st.text_area("Chapters (edit freely, one per line: time then title)",
                                  key="chapters_text", height=180)
            _items, _bad = _ch.parse_chapter_text(edited)
            for _b in _bad:
                st.warning(f"Couldn't read this line (use m:ss then a title): {_b}")
            for _p in _ch.validate(_items, st.session_state.get("chapters_total")):
                st.warning(_p)
            st.caption("Copy-ready:")
            st.code(edited, language=None)
    else:
        st.caption("Chapters appear here once the script and alignment manifest are loaded.")


with tab_thumb:
    import thumbnail_maker
    import thumbnail_studio
    import cloudflare_runner

    st.markdown("### Thumbnail")
    st.caption("1) Pick the on-image text.  2) Pick 1-3 pictures (from your video, newly generated, or your own).  "
               "3) Make thumbnails. Text is drawn by the app, so it is always spelled correctly. "
               "Output: 1280x720 JPG under 2 MB.")

    def _script_synopsis(text):
        narr = [ln.strip() for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith("[")]
        return " ".join(narr)[:1200]

    gemini_key = get_secret(["GEMINI_API_KEY"])
    gemini_model = get_secret(["GEMINI_MODEL"])
    synopsis = _script_synopsis(st.session_state.get("script_text", ""))

    # ---- scene images available from this project ----
    thumb_manifest = st.session_state.get("manifest_path")
    scene_files = {}
    if thumb_manifest and os.path.exists(thumb_manifest):
        thumb_images_dir = cloudflare_runner.get_images_dir(thumb_manifest)
        if os.path.isdir(thumb_images_dir):
            for fname in sorted(os.listdir(thumb_images_dir)):
                m = re.match(r"scene_(\d+)\.(png|jpg|jpeg)$", fname)
                if m:
                    scene_files[int(m.group(1))] = os.path.join(thumb_images_dir, fname)

    # ---- Step 1: text ----
    st.markdown("#### 1. Text on the thumbnail")
    video_title = st.text_input("Video title (paste the one you chose in Title Studio)", key="thumb_title")
    if not gemini_key:
        st.caption("No GEMINI_API_KEY in Secrets, so simple built-in ideas are used. Add the key for smarter ones.")
    if st.button("Suggest text ideas", key="thumb_suggest_text"):
        with st.spinner("Thinking of text…"):
            ideas, note = thumbnail_studio.suggest_text_ideas(video_title, synopsis, gemini_key, gemini_model)
        st.session_state["thumb_ideas"] = ideas
        st.session_state["thumb_ideas_note"] = note
        st.session_state["thumb_text_choice"] = ideas[0]["text"]   # preselect the first idea
    ideas = st.session_state.get("thumb_ideas", [])
    if st.session_state.get("thumb_ideas_note"):
        st.caption(st.session_state["thumb_ideas_note"])
    own_label = "✏️ Write my own"
    options = [i["text"] for i in ideas] + [own_label]
    choice = st.radio("Pick one", options, key="thumb_text_choice",
                      captions=[i["why"] for i in ideas] + ["Type your own text below"])
    chosen = next((i for i in ideas if i["text"] == choice), None)
    if chosen is None:
        headline = st.text_input("Your text (2-4 words)", key="thumb_own_text").strip().upper()
        keyword = st.text_input("Word(s) to show in red", key="thumb_own_kw").strip().upper()
    else:
        headline = chosen["text"]
        keyword = st.text_input("Word(s) to show in red", value=chosen["keyword"],
                                key=f"thumb_kw_{chosen['text']}").strip().upper()

    # ---- Step 2: pictures ----
    st.markdown("#### 2. Pictures")
    pool = {}      # label -> path

    cA, cB = st.columns(2)
    with cA:
        if scene_files and st.button("Find best pictures in my video", key="thumb_rank"):
            with st.spinner("Scoring your scene images…"):
                st.session_state["thumb_ranked"] = thumbnail_studio.rank_scene_images(scene_files, top_n=6)
    with cB:
        if st.button("Generate 3 new pictures", key="thumb_gen_new",
                     help="Uses about 3 images of today's Cloudflare quota"):
            accts = get_cloudflare_accounts()
            if not accts:
                st.error("No Cloudflare account found in Secrets.")
            else:
                style_name = st.session_state.get("image_style", cloudflare_runner.DEFAULT_STYLE)
                preset = cloudflare_runner.STYLE_PRESETS.get(style_name, {})
                with st.spinner("Designing three picture ideas…"):
                    concepts, cnote = thumbnail_studio.suggest_image_prompts(video_title, synopsis, gemini_key, gemini_model)
                with st.spinner("Generating on Cloudflare (about a minute)…"):
                    results, gnotes = thumbnail_studio.generate_new_pictures(
                        concepts, preset.get("prompt", ""), accts,
                        os.path.join(get_session_temp_dir(), "thumb_new"),
                        cloudflare_runner._attempt_generate, cloudflare_runner._classify_error,
                        cloudflare_runner._soften_prompt, preset.get("negative", ""))
                st.session_state["thumb_new"] = results
                st.session_state["thumb_gen_notes"] = ([cnote] if cnote else []) + gnotes
    for note in st.session_state.get("thumb_gen_notes", []):
        st.caption(note)

    for label, path in st.session_state.get("thumb_new", []):
        if os.path.exists(path):
            pool[f"New: {label}"] = path
    for n, path, score in st.session_state.get("thumb_ranked", []):
        if os.path.exists(path):
            pool[f"Scene {n} (picture score {score})"] = path
    up = st.file_uploader("Or upload your own picture", type=["png", "jpg", "jpeg"], key="thumb_upload")
    if up is not None:
        up_path = os.path.join(get_session_temp_dir(), "thumb_source_upload")
        with open(up_path, "wb") as f:
            f.write(up.read())
        pool["My upload"] = up_path
    if not pool and scene_files:
        n0 = sorted(scene_files)[0]
        pool[f"Scene {n0}"] = scene_files[n0]

    if not pool:
        st.info("No pictures yet. Generate images first, or upload one above.")
        picked = []
    else:
        labels = list(pool)
        # When the set of pictures changes (new search / new generation / upload), preselect
        # the first three; otherwise keep whatever the user ticked.
        if st.session_state.get("thumb_pool_sig") != tuple(labels):
            st.session_state["thumb_pick"] = labels[:3]
            st.session_state["thumb_pool_sig"] = tuple(labels)
        else:
            st.session_state["thumb_pick"] = [l for l in st.session_state.get("thumb_pick", []) if l in labels]
        picked = st.multiselect("Pictures to use (up to 3 recommended)", labels, key="thumb_pick")
        cols = st.columns(min(3, max(1, len(labels))))
        for k, label in enumerate(labels[:9]):
            with cols[k % len(cols)]:
                st.image(pool[label], caption=label, width="stretch")

    # ---- Step 3: render ----
    st.markdown("#### 3. Make thumbnails")
    layout_names = {"overlay": "Picture full-frame, text on dark fade",
                    "split_left": "Text left, picture right",
                    "split_right": "Picture left, text right"}
    layouts = st.multiselect("Layouts", list(layout_names), default=["overlay"],
                             format_func=lambda k: layout_names[k], key="thumb_layouts")
    punch = st.checkbox("Boost contrast and colour slightly", value=True, key="thumb_punch")

    if st.button("Make thumbnails", type="primary", key="thumb_make"):
        if not picked:
            st.error("Pick at least one picture.")
        elif not headline:
            st.error("Choose or write the thumbnail text.")
        elif not layouts:
            st.error("Pick at least one layout.")
        elif len(picked) * len(layouts) > 6:
            st.error("That would make more than 6 thumbnails. Pick fewer pictures or layouts.")
        else:
            made, errors = [], []
            tdir = os.path.join(get_session_temp_dir(), "thumb_out")
            os.makedirs(tdir, exist_ok=True)
            for pi, label in enumerate(picked, 1):
                src = pool[label]
                try:
                    if punch:
                        src = thumbnail_studio.enhance_picture(src, os.path.join(tdir, f"src_{pi}.png"))
                except Exception:
                    src = pool[label]
                for layout in layouts:
                    out = os.path.join(tdir, f"thumbnail_{pi}_{layout}.jpg")
                    try:
                        thumbnail_maker.make_thumbnail(src, headline, keyword, out, layout)
                        made.append({"label": label, "layout": layout, "path": out,
                                     "warns": thumbnail_studio.quality_report(out, headline)})
                        upload_to_project(out, f"thumbnails/{os.path.basename(out)}")
                    except Exception as e:
                        errors.append(f"{label} / {layout}: {e}")
            st.session_state["thumb_results"] = made
            for e in errors:
                st.error(f"Thumbnail failed: {e}")

    for k, item in enumerate(st.session_state.get("thumb_results", [])):
        if os.path.exists(item["path"]):
            big, small = st.columns([4, 1])
            with big:
                st.image(item["path"], caption=f"{item['label']} · {layout_names.get(item['layout'], item['layout'])}",
                         width="stretch")
            with small:
                st.caption("Phone size")
                st.image(item["path"], width=168)
            for w in item["warns"]:
                st.warning(w)
            with open(item["path"], "rb") as f:
                st.download_button(f"Download thumbnail {k + 1}", f,
                                   file_name=f"thumbnail_{k + 1}_{item['layout']}.jpg",
                                   mime="image/jpeg", key=f"dl_thumb_{k}")

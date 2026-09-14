"""
video_assembler.py
Assembles the final video from generated scene images + narration audio,
timed against scene_manifest.csv (start_time/end_time columns).

Two design decisions worth knowing about:

1. GAP FILLING, NOT SKIPPING: if a scene's image isn't ready yet, its slot
   is NOT dropped from the timeline — dropping it would shrink the video's
   total length and throw every later scene out of sync with the (unchanged)
   audio track. Instead the nearest available neighboring scene's image is
   reused for that duration, and the scene is reported back so the real
   artwork can be swapped in once it exists.

2. MOTION INSTEAD OF A DURATION CAP: every scene image gets a slow,
   alternating zoom-in/zoom-out (Ken Burns effect) for its exact narrated
   duration, rather than capping how long an image can stay on screen.
   A hard cap (e.g. 5s max) would force the picture to stop matching the
   narration for whatever's left of that scene — motion solves "boring"
   without ever breaking audio/video sync.

Also reconciles the total image timeline to the actual narration length
via ffprobe, so the video is never a hair short or long of the audio.

Requires the 'ffmpeg' and 'ffprobe' system binaries (installed together by
the same apt package — see packages.txt). Not bundled with the base image
on Streamlit Community Cloud without it.
"""

import os
import subprocess
import tempfile

VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 1024
FPS = 30

# How far in/out the Ken Burns effect zooms over a scene's duration.
MAX_ZOOM = 1.15


def _hhmmss_to_seconds(value):
    if not value or not isinstance(value, str):
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None


def _load_manifest_df(manifest_path):
    import pandas as pd

    try:
        df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(manifest_path)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "").str.lower()

    # Match pollinations_runner's scene_id coercion exactly so both modules
    # agree on numbering even if the manifest has blank/non-numeric rows.
    if "scene_id" in df.columns:
        fallback_series = pd.Series(range(1, len(df) + 1), index=df.index)
        df["scene_id"] = pd.to_numeric(df["scene_id"], errors="coerce").fillna(fallback_series).astype(int)
    else:
        df["scene_id"] = list(range(1, len(df) + 1))

    return df


def _get_audio_duration_seconds(audio_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, TypeError, OSError):
        return None


def _render_scene_clip(image_path, duration, output_path, zoom_out):
    """
    Renders one scene's image into a short video clip with a slow,
    continuous zoom (Ken Burns effect) — zooming in for even-indexed
    scenes, out for odd-indexed ones, so consecutive scenes don't feel
    repetitive. Kept centered so it never crops off anything important.
    """
    duration = max(0.5, duration)
    frames = max(1, round(duration * FPS))
    step = (MAX_ZOOM - 1.0) / frames

    if zoom_out:
        zoom_expr = f"if(eq(on,1),{MAX_ZOOM},max(zoom-{step:.6f},1.0))"
    else:
        zoom_expr = f"min(zoom+{step:.6f},{MAX_ZOOM})"

    vf = (
        f"zoompan=z='{zoom_expr}':d={frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={FPS}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "faster", "-crf", "18",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (scene clip render) failed:\n{result.stderr[-1500:]}")


def assemble_video(manifest_path, images_dir, audio_path, output_path, progress_callback=None):
    """
    progress_callback(scenes_processed, total_scenes, message)

    Returns (output_path, filled_scene_ids). filled_scene_ids lists scenes
    whose own image wasn't ready, so a neighboring scene's image was used
    to keep the timeline complete and in sync — reassemble later once those
    images exist to swap in the real artwork.
    """
    df = _load_manifest_df(manifest_path)
    total = len(df)

    scene_entries = []
    for i, row in df.iterrows():
        try:
            scene_id = int(row["scene_id"])
        except Exception:
            scene_id = i + 1

        image_path = os.path.join(images_dir, f"scene_{scene_id:03d}.png")
        if not os.path.exists(image_path):
            image_path = None

        start_s = _hhmmss_to_seconds(row.get("start_time"))
        end_s = _hhmmss_to_seconds(row.get("end_time"))
        if start_s is not None and end_s is not None and end_s > start_s:
            duration = end_s - start_s
        else:
            duration = 3.0  # fallback when timing is missing

        scene_entries.append({"scene_id": scene_id, "image_path": image_path, "duration": duration})

    # Forward-fill gaps from the previous available scene, then back-fill
    # any leading gaps (before the first available image) from the next one.
    last_seen = None
    for entry in scene_entries:
        if entry["image_path"] is not None:
            last_seen = entry["image_path"]
        elif last_seen is not None:
            entry["image_path"] = last_seen
            entry["filled_from_adjacent"] = True

    next_seen = None
    for entry in reversed(scene_entries):
        if entry["image_path"] is not None and not entry.get("filled_from_adjacent"):
            next_seen = entry["image_path"]
        elif entry["image_path"] is None and next_seen is not None:
            entry["image_path"] = next_seen
            entry["filled_from_adjacent"] = True

    if all(e["image_path"] is None for e in scene_entries):
        raise RuntimeError("No scene images found — generate images before assembling the video.")

    filled_scene_ids = [e["scene_id"] for e in scene_entries if e.get("filled_from_adjacent")]

    # Reconcile the image timeline to the actual audio length so the video
    # is never a hair short or long of the narration — no manual checking
    # required. Correct the last scene first (least visible change); if
    # that would make it awkwardly short, spread the correction evenly
    # instead.
    audio_duration = _get_audio_duration_seconds(audio_path)
    if audio_duration is not None:
        total_image_duration = sum(e["duration"] for e in scene_entries)
        diff = audio_duration - total_image_duration
        if abs(diff) > 0.3 and scene_entries:
            adjusted_last = scene_entries[-1]["duration"] + diff
            if adjusted_last >= 0.5:
                scene_entries[-1]["duration"] = adjusted_last
            else:
                per_scene_adjust = diff / len(scene_entries)
                for entry in scene_entries:
                    entry["duration"] = max(0.5, entry["duration"] + per_scene_adjust)
            if progress_callback:
                progress_callback(0, total,
                                   f"Adjusted scene timing by {diff:+.2f}s total to exactly match narration length.")

    work_dir = tempfile.mkdtemp(prefix="video_assembly_")
    clip_paths = []

    for i, entry in enumerate(scene_entries):
        clip_path = os.path.join(work_dir, f"clip_{i:04d}.mp4")
        zoom_out = (i % 2 == 1)
        _render_scene_clip(entry["image_path"], entry["duration"], clip_path, zoom_out)
        clip_paths.append(clip_path)

        if progress_callback:
            note = " (filled from adjacent scene — swap in later)" if entry.get("filled_from_adjacent") else ""
            progress_callback(i + 1, total, f"Scene {entry['scene_id']} rendered ({entry['duration']:.1f}s){note}.")

    concat_list_path = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        f.write("\n".join(f"file '{p}'" for p in clip_paths))

    silent_video_path = os.path.join(work_dir, "silent.mp4")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy",
        silent_video_path,
    ]
    result = subprocess.run(cmd_concat, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (clip concat) failed:\n{result.stderr[-2000:]}")

    cmd_mux = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd_mux, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (audio mux) failed:\n{result.stderr[-2000:]}")

    return output_path, filled_scene_ids

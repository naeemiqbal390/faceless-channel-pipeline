"""
video_assembler.py
Assembles the final video from generated scene images + narration audio,
timed against scene_manifest.csv (start_time/end_time columns).

If a scene's image isn't ready yet, its slot is NOT dropped from the
timeline — dropping it would shrink the video's total length and throw
every later scene out of sync with the (unchanged) audio track. Instead
the nearest available neighboring scene's image is reused for that
duration, and the scene is reported back so it can be swapped in for
real once its image finishes generating.

Requires the 'ffmpeg' system binary. On Streamlit Community Cloud this means
adding a packages.txt file (see accompanying packages.txt) so the platform
apt-installs it before the app starts — it is not bundled with the base image.
"""

import os
import subprocess
import tempfile


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

    # Match gemini_runner's scene_id coercion exactly so both modules agree
    # on numbering even if the manifest has blank/non-numeric rows.
    if "scene_id" in df.columns:
        fallback_series = pd.Series(range(1, len(df) + 1), index=df.index)
        df["scene_id"] = pd.to_numeric(df["scene_id"], errors="coerce").fillna(fallback_series).astype(int)
    else:
        df["scene_id"] = list(range(1, len(df) + 1))

    return df


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

    work_dir = tempfile.mkdtemp(prefix="video_assembly_")
    concat_list_path = os.path.join(work_dir, "concat_list.txt")
    concat_lines = []
    last_image_line = None

    for i, entry in enumerate(scene_entries):
        if entry["image_path"] is None:
            continue  # only possible if truly no image exists in the whole batch
        image_line = f"file '{entry['image_path']}'"
        concat_lines.append(image_line)
        concat_lines.append(f"duration {entry['duration']}")
        last_image_line = image_line

        if progress_callback:
            note = " (filled from adjacent scene — swap in later)" if entry.get("filled_from_adjacent") else ""
            progress_callback(i + 1, total, f"Scene {entry['scene_id']} queued ({entry['duration']:.1f}s){note}.")

    # The ffmpeg concat demuxer requires the final image repeated without a
    # trailing duration line, or the last image gets cut short.
    concat_lines.append(last_image_line)

    with open(concat_list_path, "w") as f:
        f.write("\n".join(concat_lines))

    silent_video_path = os.path.join(work_dir, "silent.mp4")
    cmd_video = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-vsync", "vfr", "-r", "30", "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        silent_video_path,
    ]
    result = subprocess.run(cmd_video, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (image assembly) failed:\n{result.stderr[-2000:]}")

    cmd_mux = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd_mux, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (audio mux) failed:\n{result.stderr[-2000:]}")

    return output_path, filled_scene_ids

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

2. MOTION INSTEAD OF A DURATION CAP: every scene image gets exactly ONE
   slow, small movement (zoom in, zoom out, or a gentle pan left/right/up/
   down, cycling in a fixed varied order) for its exact narrated duration,
   and consecutive images are joined with a slow cross-dissolve, rather than
   capping how long an image can stay on screen.
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


# Constant zoom speed (fraction per second) rather than a fixed total zoom
# amount — a 2-second scene and a 20-second scene zooming to the same total
# 15% would make the short one look like a snap-jump, not a slow drift.
ZOOM_RATE_PER_SECOND = 0.015

# Upscale factor applied before zoompan. Zooming directly on a 1024x1024
# source gives the crop window too little sub-pixel precision, so it snaps
# between slightly different integer pixel positions each frame — visible
# as a jittery "shake" rather than a smooth drift. Pre-scaling to a much
# larger canvas first gives zoompan the precision to move smoothly.
ZOOMPAN_UPSCALE = 4096


# One movement per image, cycling so neighbours never repeat the same move.
MOTION_CYCLE = ["zoom_in", "pan_right", "zoom_out", "pan_left",
                "zoom_in", "pan_down", "zoom_out", "pan_up"]

# Zoom level used while panning. 1.09 means the picture is 9% larger than the
# frame, so a pan travels about 8% of the width: slow and small.
PAN_ZOOM = 1.09

# Length of the cross-dissolve between two images (seconds). Capped for short
# scenes so a dissolve never takes more than ~30% of either neighbour.
TRANSITION_SECONDS = 0.8
MIN_CLIP_FRAMES = 6


def _motion_exprs(motion, offset_frames, total_frames, length_seconds):
    """zoompan z/x/y expressions for ONE movement. Progress p runs 0 -> 1 over
    the image's whole on-screen life (including the part shared with a
    dissolve), so the movement stays continuous across the dissolve."""
    total_frames = max(1, int(total_frames))
    p = f"min(1,(on-1+{int(offset_frames)})/{total_frames})"
    amount = min(MAX_ZOOM - 1.0, ZOOM_RATE_PER_SECOND * max(0.5, length_seconds))
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"
    if motion == "zoom_in":
        return f"1+{amount:.5f}*{p}", cx, cy
    if motion == "zoom_out":
        return f"1+{amount:.5f}*(1-{p})", cx, cy
    z = f"{PAN_ZOOM}"
    if motion == "pan_right":
        return z, f"(iw-iw/zoom)*{p}", cy
    if motion == "pan_left":
        return z, f"(iw-iw/zoom)*(1-{p})", cy
    if motion == "pan_down":
        return z, cx, f"(ih-ih/zoom)*{p}"
    if motion == "pan_up":
        return z, cx, f"(ih-ih/zoom)*(1-{p})"
    return "1", cx, cy


def _zoompan_chain(frames, motion, offset_frames, total_frames):
    z, x, y = _motion_exprs(motion, offset_frames, total_frames, total_frames / FPS)
    return (
        f"scale={ZOOMPAN_UPSCALE}:{ZOOMPAN_UPSCALE}:flags=lanczos,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}"
        f":s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={FPS},setsar=1,format=yuv420p"
    )


def _render_scene_clip(image_path, frames, motion, in_frames, total_frames, output_path,
                       next_image_path=None, tail_frames=0, next_motion=None, next_total_frames=0):
    """
    Renders one scene's clip of exactly `frames` frames.

    The clip shows its own image with one movement. During its last
    `tail_frames` frames it dissolves into the NEXT image, whose own movement
    has already begun (so it keeps moving smoothly after the cut point,
    where the next clip takes over without any jump).

    in_frames    - frames of this image already shown during the previous
                   clip's dissolve (movement progress starts there)
    total_frames - this image's whole on-screen life in frames
    """
    cur = _zoompan_chain(frames, motion, in_frames, total_frames)

    if next_image_path and tail_frames > 0:
        nxt = _zoompan_chain(tail_frames, next_motion, 0, next_total_frames)
        t_s = tail_frames / FPS
        off_s = (frames - tail_frames) / FPS
        graph = (
            f"[0:v]{cur}[a];[1:v]{nxt}[b];"
            f"[a][b]xfade=transition=fade:duration={t_s:.4f}:offset={off_s:.4f},"
            f"tpad=stop=3:stop_mode=clone[v]"
        )
        inputs = ["-i", image_path, "-i", next_image_path]
    else:
        graph = f"[0:v]{cur},tpad=stop=3:stop_mode=clone[v]"
        inputs = ["-i", image_path]

    cmd = (["ffmpeg", "-y"] + inputs +
           ["-filter_complex", graph, "-map", "[v]",
            "-frames:v", str(frames), "-r", str(FPS),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "faster", "-crf", "18",
            output_path])
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
    audio_duration = _get_audio_duration_seconds(audio_path)

    scene_entries = []
    for i, row in df.iterrows():
        try:
            scene_id = int(row["scene_id"])
        except Exception:
            scene_id = i + 1

        image_path = os.path.join(images_dir, f"scene_{scene_id:03d}.png")
        if not os.path.exists(image_path):
            image_path = None

        scene_entries.append({
            "scene_id": scene_id,
            "image_path": image_path,
            "start_s": _hhmmss_to_seconds(row.get("start_time")),
            "end_s": _hhmmss_to_seconds(row.get("end_time")),
        })

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

    # TIMING: each scene's image covers from ITS OWN start_time up to the
    # NEXT scene's start_time — NOT each scene's own (end_time - start_time)
    # span. Real manifests routinely have several-second gaps between one
    # scene's detected end and the next scene's detected start (a pause, a
    # beat, alignment slack). Chaining only each scene's narrow span
    # together silently drops every one of those gaps — the video ends up
    # shorter than the narration and drifts further out of sync with every
    # scene after the first gap. Using start-to-next-start guarantees the
    # visual timeline covers the ENTIRE audio with zero gaps.
    starts = [e["start_s"] for e in scene_entries]
    for i, s in enumerate(starts):
        if s is None:
            if i == 0:
                starts[i] = 0.0
            else:
                fallback = scene_entries[i].get("end_s")
                prev = starts[i - 1]
                starts[i] = fallback if (fallback is not None and fallback > prev) else prev + 0.5

    # Guard against any out-of-order or noisy timestamps in the manifest.
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i - 1]

    final_end = audio_duration
    if final_end is None or final_end < starts[-1]:
        final_end = starts[-1] + (scene_entries[-1].get("end_s") or starts[-1] + 3.0) - starts[-1]

    boundaries = starts + [final_end]
    # The first scene covers from t=0, not from its own detected start —
    # there's usually a few seconds of intro narration before the first
    # scene's exact words begin, and skipping that would leave the video's
    # opening seconds blank.
    boundaries[0] = 0.0

    # Frame-exact timeline: every clip length is a whole number of frames and
    # the clip boundaries come from rounding the ABSOLUTE times, so rounding
    # errors never accumulate (zero drift against the audio over any length).
    fb = [int(round(t * FPS)) for t in boundaries]
    for i in range(1, len(fb)):
        if fb[i] < fb[i - 1] + MIN_CLIP_FRAMES:
            fb[i] = fb[i - 1] + MIN_CLIP_FRAMES
    frames_list = [fb[i + 1] - fb[i] for i in range(len(scene_entries))]
    for i, entry in enumerate(scene_entries):
        entry["duration"] = frames_list[i] / FPS

    # Dissolve length at each boundary (frames); 0 = hard cut for tiny clips.
    n_scenes = len(scene_entries)
    base_tf = int(round(TRANSITION_SECONDS * FPS))
    tf = []
    for i in range(n_scenes - 1):
        t = min(base_tf, int(0.3 * min(frames_list[i], frames_list[i + 1])))
        tf.append(t if t >= 3 else 0)
    t_in = [0] + tf                      # frames of image i shown inside the previous dissolve
    life = [frames_list[i] + t_in[i] for i in range(n_scenes)]   # image i's whole on-screen life

    if progress_callback:
        total_span = boundaries[-1] - boundaries[0]
        audio_note = f", audio is {audio_duration:.1f}s" if audio_duration is not None else " (audio duration unknown — using manifest timing as-is)"
        progress_callback(0, total, f"Scene timeline covers {total_span:.1f}s gaplessly{audio_note}.")

    work_dir = tempfile.mkdtemp(prefix="video_assembly_")
    clip_paths = []

    for i, entry in enumerate(scene_entries):
        clip_path = os.path.join(work_dir, f"clip_{i:04d}.mp4")
        motion = MOTION_CYCLE[i % len(MOTION_CYCLE)]
        has_next = i < n_scenes - 1 and tf[i] > 0
        _render_scene_clip(
            entry["image_path"], frames_list[i], motion, t_in[i], life[i], clip_path,
            next_image_path=scene_entries[i + 1]["image_path"] if has_next else None,
            tail_frames=tf[i] if has_next else 0,
            next_motion=MOTION_CYCLE[(i + 1) % len(MOTION_CYCLE)] if has_next else None,
            next_total_frames=life[i + 1] if has_next else 0,
        )
        clip_paths.append(clip_path)

        if progress_callback:
            note = " (filled from adjacent scene — swap in later)" if entry.get("filled_from_adjacent") else ""
            progress_callback(i + 1, total, f"Scene {entry['scene_id']} rendered ({entry['duration']:.1f}s, {motion.replace('_', ' ')}){note}.")

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

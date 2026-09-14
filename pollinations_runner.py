"""
pollinations_runner.py
Image generation via Pollinations.ai's image API.

- Plain HTTP GET to https://image.pollinations.ai/prompt/{prompt}, returns
  raw image bytes directly — no SDK. Works with no API key at all, but a
  free Pollinations account token (from auth.pollinations.ai, no card)
  unlocks: a faster rate limit, a genuinely watermark-free "nologo" result,
  and access to premium models paid for out of a small weekly free credit
  ("Pollen") allowance.
- MODEL HIERARCHY: tries the best available model first (premium models
  that spend Pollen credits), and automatically falls back down the list
  the moment a model is unavailable or its credit is exhausted — landing
  on "flux"/"turbo", which are genuinely free and unlimited, for the bulk
  of any run once premium credit runs out. This means the first handful of
  images in a run may be noticeably higher quality than the rest — that's
  expected, not a bug, given how small the free weekly Pollen allowance is.
- Same resumable / non-blocking-failure / automatic-retry-round design as
  before: a failed scene is parked and retried later without blocking the
  scenes after it, and successes always land under their correct scene_id
  filename so they slot into the right spot in the video.
"""

import os
import time
import json
import random
import hashlib
import tempfile
import shutil
from urllib.parse import quote

RUNS_DIR = os.path.join(tempfile.gettempdir(), "pipeline_runs")

BASE_URL = "https://image.pollinations.ai/prompt/"

# Best-quality-first. Premium models (💎 in Pollinations' own docs) spend a
# small weekly free Pollen credit and require a token; "flux"/"turbo" are
# the always-free, unlimited backbone at the bottom of the list. The run
# tries top-down and permanently drops to the next candidate the moment a
# model reports itself unavailable or out of credit — add new premium
# model names here as Pollinations ships them, no other code changes needed.
MODEL_CANDIDATES_WITH_TOKEN = ["nanobanana", "seedream", "gptimage", "flux", "turbo"]
MODEL_CANDIDATES_NO_TOKEN = ["flux", "turbo"]  # premium models require a token — skip straight to the free ones

RETRY_ROUNDS = 3

# Anonymous Pollinations use is documented at roughly one request per 15
# seconds. A registered (free, no-card) account raises this meaningfully —
# since a token is now expected to be configured, pacing defaults to the
# faster registered-tier assumption and only falls back to the slower
# anonymous pacing if no token is present.
ANON_PACING = dict(floor_rpm=2, ceiling_rpm=5, start_rpm=4)
TOKEN_PACING = dict(floor_rpm=4, ceiling_rpm=15, start_rpm=8)

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# A real image is never this small — anything under this is almost
# certainly an HTML error page or truncated response, not a photo.
MIN_VALID_IMAGE_BYTES = 2000

# Models whose "quality" parameter Pollinations documents as actually
# honored (others are said to ignore it, so there's no harm sending it
# broadly, but this is what it's known to matter for).
QUALITY_AWARE_MODELS = {"gptimage", "gptimage-large", "gpt-image-2"}

# Visual style presets — prepended to every scene prompt. "Style" here means
# a general aesthetic descriptor (palette, line work, era), not a request to
# reproduce any specific studio's copyrighted characters or film stills.
STYLE_PRESETS = {
    "Stick Figure": (
        "Simple black and white stick-figure illustration, minimal line art "
        "style, plain background. Scene: "
    ),
    "Anime / Hand-Painted (Ghibli-inspired)": (
        "Hand-painted 2D anime background art, soft watercolor palette, "
        "whimsical storybook atmosphere, warm natural lighting. Scene: "
    ),
    "1980s Retro Anime": (
        "1980s retro anime style, grainy film texture, bold cel-shaded "
        "colors, vintage VHS aesthetic. Scene: "
    ),
    "Watercolor": (
        "Soft watercolor painting, gentle visible brush strokes, muted "
        "pastel color palette. Scene: "
    ),
    "Comic Book": (
        "Bold comic book illustration, heavy ink outlines, halftone "
        "shading, vibrant saturated colors. Scene: "
    ),
    "Photorealistic": (
        "Photorealistic, cinematic lighting, high detail, shot on 35mm "
        "film. Scene: "
    ),
}
DEFAULT_STYLE = "Stick Figure"


# --------------------------------------------------------------------------
# Paths / manifest helpers
# --------------------------------------------------------------------------

def compute_run_id(manifest_path):
    try:
        with open(manifest_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "default_run"


def get_images_dir(manifest_path):
    run_id = compute_run_id(manifest_path)
    images_dir = os.path.join(RUNS_DIR, run_id, "scene_images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def _load_manifest_df(manifest_path):
    import pandas as pd

    try:
        df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(manifest_path)

    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "").str.lower()

    if "scene_id" not in df.columns:
        df["scene_id"] = list(range(1, len(df) + 1))
    if "prompt" not in df.columns:
        df["prompt"] = "stick figure drawing"

    fallback_series = pd.Series(range(1, len(df) + 1), index=df.index)
    df["scene_id"] = pd.to_numeric(df["scene_id"], errors="coerce").fillna(fallback_series).astype(int)

    return df


def get_resume_status(manifest_path):
    try:
        df = _load_manifest_df(manifest_path)
        images_dir = get_images_dir(manifest_path)
        total_images = len(df)
        done = 0
        for _, row in df.iterrows():
            try:
                scene_id = int(row["scene_id"])
            except Exception:
                continue
            if os.path.exists(os.path.join(images_dir, f"scene_{scene_id:03d}.png")):
                done += 1
        return {
            "images_dir": images_dir,
            "total_images": total_images,
            "done_images": done,
            "has_progress": done > 0,
        }
    except Exception:
        return {
            "images_dir": "",
            "total_images": 0,
            "done_images": 0,
            "has_progress": False,
        }


# --------------------------------------------------------------------------
# Adaptive pacing
# --------------------------------------------------------------------------

class AdaptiveRateLimiter:
    """
    Paces requests between a floor and ceiling requests-per-minute, easing
    the interval down on sustained success and snapping it back up the
    moment a rate-limit response is seen.
    """

    def __init__(self, floor_rpm, ceiling_rpm, start_rpm, successes_per_speedup=4):
        self.min_interval = 60.0 / ceiling_rpm
        self.max_interval = 60.0 / floor_rpm
        self.interval = 60.0 / start_rpm
        self._consecutive_successes = 0
        self._successes_per_speedup = successes_per_speedup

    def wait(self):
        time.sleep(self.interval)

    def record_success(self):
        self._consecutive_successes += 1
        if self._consecutive_successes >= self._successes_per_speedup:
            self._consecutive_successes = 0
            self.interval = max(self.min_interval, self.interval * 0.85)

    def record_rate_limit(self):
        self._consecutive_successes = 0
        self.interval = min(self.max_interval, self.interval * 1.8)

    @property
    def current_rpm(self):
        return round(60.0 / self.interval, 1)


# --------------------------------------------------------------------------
# Model selection / fallback
# --------------------------------------------------------------------------

class ModelState:
    def __init__(self, candidates):
        self.candidates = candidates
        self.index = 0

    def current(self):
        return self.candidates[self.index]

    def advance(self):
        if self.index < len(self.candidates) - 1:
            self.index += 1
            return True
        return False


def _classify_error(err_text):
    lowered = err_text.lower()
    # Out of Pollen credit, or this model needs a higher account tier than
    # this token has — permanent for the rest of this run, so treat it the
    # same as model-unavailable: drop to the next candidate immediately.
    if any(tok in lowered for tok in
           ["402", "insufficient", "credit", "pollen", "tier required", "upgrade"]):
        return "model_unavailable"
    if "429" in err_text or "too many requests" in lowered or "rate limit" in lowered:
        return "rate_limit"
    if any(tok in lowered for tok in ["502", "503", "504", "bad gateway", "gateway timeout", "unavailable"]):
        return "rate_limit"  # treat server overload the same as rate limiting — back off, don't hammer
    if any(tok in lowered for tok in ["404", "model not found", "unsupported model", "invalid model"]):
        return "model_unavailable"
    return "other"


def _attempt_generate(model_name, prompt_text, out_path, token=None):
    import requests

    url = BASE_URL + quote(prompt_text)
    params = {
        "model": model_name,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "seed": random.randint(1, 2_147_483_647),
        "nologo": "true",
        "private": "true",
        "safe": "true",
    }
    if model_name in QUALITY_AWARE_MODELS:
        params["quality"] = "high"
    if token:
        params["token"] = token

    response = requests.get(url, params=params, timeout=90)

    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text[:300]}")

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(f"non-image response (content-type={content_type}): {response.text[:300]}")

    if len(response.content) < MIN_VALID_IMAGE_BYTES:
        raise RuntimeError(f"response too small to be a real image ({len(response.content)} bytes)")

    with open(out_path, "wb") as f:
        f.write(response.content)
    return True


def _generate_with_fallback(model_state, prompt_text, out_path, rate_limiter, token):
    """
    Single "best effort" attempt at one image: tries the current model
    (with one same-model retry for rate-limit/transient hiccups), and
    switches to the next candidate model when the failure looks like a
    genuine availability or credit problem. Raises on total failure — the
    caller decides what happens next (park it for a later retry round).
    """
    last_err = None

    for _ in range(len(model_state.candidates)):
        model_name = model_state.current()

        for local_attempt in range(2):
            try:
                _attempt_generate(model_name, prompt_text, out_path, token)
                rate_limiter.record_success()
                return
            except Exception as e:
                err_text = str(e)
                kind = _classify_error(err_text)
                last_err = err_text

                if kind == "model_unavailable":
                    break
                elif kind == "rate_limit":
                    rate_limiter.record_rate_limit()
                    if local_attempt == 0:
                        time.sleep(rate_limiter.interval)
                        continue
                    break
                else:
                    if local_attempt == 0:
                        time.sleep(3)
                        continue
                    break

        if _classify_error(last_err or "") == "model_unavailable":
            if not model_state.advance():
                break
        else:
            break

    raise RuntimeError(last_err or "Unknown image generation error")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_image_generation(manifest_path, api_key, progress_callback=None, style=DEFAULT_STYLE):
    """
    Generates one image per manifest row via Pollinations.ai.

    'api_key' is an OPTIONAL Pollinations account token — pass "" or None to
    use anonymously (free models only, slower pacing). With a token, premium
    models are tried first and pacing assumes the faster registered tier.

    'style' selects a STYLE_PRESETS key to prepend to every scene prompt.

    progress_callback(done_images, total_images, message)

    Returns (images_dir, zip_path, failed_scene_ids).
    """
    token = api_key or None
    style_prefix = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])

    df = _load_manifest_df(manifest_path)
    images_dir = get_images_dir(manifest_path)

    total_images = len(df)
    done_images = 0
    pending = []

    for _, row in df.iterrows():
        try:
            scene_id = int(row["scene_id"])
        except Exception:
            continue
        out_path = os.path.join(images_dir, f"scene_{scene_id:03d}.png")
        prompt = str(row["prompt"]) if row["prompt"] == row["prompt"] else "stick figure drawing"
        if os.path.exists(out_path):
            done_images += 1
        else:
            pending.append((scene_id, prompt, out_path))

    if progress_callback and done_images:
        progress_callback(done_images, total_images,
                           f"Resuming — {done_images}/{total_images} images already done.")

    candidates = MODEL_CANDIDATES_WITH_TOKEN if token else MODEL_CANDIDATES_NO_TOKEN
    pacing = TOKEN_PACING if token else ANON_PACING
    model_state = ModelState(list(candidates))
    rate_limiter = AdaptiveRateLimiter(**pacing)
    last_model_reported = model_state.current()

    if progress_callback and pending:
        note = "" if token else " (anonymous — add a Pollinations token for premium models + faster pacing)"
        progress_callback(done_images, total_images, f"Starting with model: {model_state.current()}{note}")

    def _run_one(scene_id, prompt, out_path):
        nonlocal done_images, last_model_reported
        full_prompt = style_prefix + prompt
        rate_limiter.wait()
        try:
            _generate_with_fallback(model_state, full_prompt, out_path, rate_limiter, token)
            done_images += 1
            if progress_callback:
                progress_callback(done_images, total_images,
                                   f"Scene {scene_id} done (~{rate_limiter.current_rpm} req/min).")
            if model_state.current() != last_model_reported:
                last_model_reported = model_state.current()
                if progress_callback:
                    progress_callback(done_images, total_images,
                                       f"Switched to fallback image model: {last_model_reported}")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(done_images, total_images, f"Scene {scene_id} failed, parked for retry: {e}")
            return False

    # Main pass — one shot per scene, in order. A failure is parked, not
    # retried in place, so it never holds up the scenes after it.
    failed = []
    for scene_id, prompt, out_path in pending:
        if not _run_one(scene_id, prompt, out_path):
            failed.append((scene_id, prompt, out_path))

    # Automatic retry rounds over whatever's still missing.
    round_num = 1
    while failed and round_num <= RETRY_ROUNDS:
        if progress_callback:
            progress_callback(done_images, total_images,
                               f"Retry round {round_num}/{RETRY_ROUNDS} — {len(failed)} scene(s) remaining.")
        still_failed = []
        for scene_id, prompt, out_path in failed:
            if not _run_one(scene_id, prompt, out_path):
                still_failed.append((scene_id, prompt, out_path))
        failed = still_failed
        round_num += 1

    failed_scene_ids = [f[0] for f in failed]

    run_dir = os.path.dirname(images_dir)
    if failed_scene_ids:
        try:
            with open(os.path.join(run_dir, "failed_scenes.json"), "w") as f:
                json.dump(failed_scene_ids, f)
        except Exception:
            pass

    zip_base = os.path.join(run_dir, "scene_images_batch")
    zip_path = shutil.make_archive(zip_base, "zip", images_dir)

    return images_dir, zip_path, failed_scene_ids

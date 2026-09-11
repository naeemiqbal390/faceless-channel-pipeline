"""
pollinations_runner.py
Image generation via Pollinations.ai's free, keyless image API.

- Plain HTTP GET to https://image.pollinations.ai/prompt/{prompt}, returns
  raw image bytes directly — no SDK, no API key required for anonymous use.
- No published daily cap (unlike Gemini's free tier, which turned out to
  have zero allocation for image models on this account) — it's throttled
  by request rate instead, roughly one request per ~15 seconds anonymously.
  A free account (no card) raises that rate and drops the watermark.
- Same resumable / non-blocking-failure / automatic-retry-round design as
  the Gemini version: a failed scene is parked and retried later without
  blocking the scenes after it, and successes always land under their
  correct scene_id filename so they slot into the right spot in the video.
- Falls back from "flux" to "turbo" if the primary model has trouble.
"""

import os
import re
import time
import json
import random
import hashlib
import tempfile
import shutil
from urllib.parse import quote

RUNS_DIR = os.path.join(tempfile.gettempdir(), "pipeline_runs")

BASE_URL = "https://image.pollinations.ai/prompt/"

# Best-quality-first. "turbo" is the fallback if "flux" has trouble — still
# decent quality, just faster/lighter. Add new Pollinations model names here
# if they add better free models later.
MODEL_CANDIDATES = ["flux", "turbo"]

RETRY_ROUNDS = 3

# Anonymous Pollinations use is documented at roughly one request per 15
# seconds. A free (no-card) account at auth.pollinations.ai raises this —
# if you register, lower ceiling_rpm's denominator accordingly, i.e. raise
# ceiling_rpm below.
FLOOR_RPM = 2      # slowest: one request per 30s, if things are struggling
CEILING_RPM = 5    # fastest: one request per 12s, a bit above the documented anon rate
START_RPM = 4

# Prepended to every scene prompt to keep the established visual style
# consistent across the whole batch.
STYLE_PREFIX = (
    "Simple black and white stick-figure illustration, minimal line art "
    "style, plain background. Scene: "
)

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# A real image is never this small — anything under this is almost
# certainly an HTML error page or truncated response, not a photo.
MIN_VALID_IMAGE_BYTES = 2000


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

    def __init__(self, floor_rpm=FLOOR_RPM, ceiling_rpm=CEILING_RPM,
                 start_rpm=START_RPM, successes_per_speedup=4):
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
    genuine model-availability problem. Raises on total failure — the
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

def run_image_generation(manifest_path, api_key, progress_callback=None):
    """
    Generates one image per manifest row via Pollinations.ai.

    'api_key' here is an OPTIONAL Pollinations account token (unlike Gemini,
    Pollinations works fully anonymously — pass "" or None to use it that
    way). A token just raises the rate limit and drops the watermark.

    progress_callback(done_images, total_images, message)

    Returns (images_dir, zip_path, failed_scene_ids).
    """
    token = api_key or None

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

    model_state = ModelState(list(MODEL_CANDIDATES))
    rate_limiter = AdaptiveRateLimiter()
    last_model_reported = model_state.current()

    if progress_callback and pending:
        note = " (anonymous — pass a token for a faster rate)" if not token else ""
        progress_callback(done_images, total_images, f"Starting with model: {model_state.current()}{note}")

    def _run_one(scene_id, prompt, out_path):
        nonlocal done_images, last_model_reported
        full_prompt = STYLE_PREFIX + prompt
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

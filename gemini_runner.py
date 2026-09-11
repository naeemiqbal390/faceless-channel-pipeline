"""
gemini_runner.py
Image generation via Google's Gemini image models, replacing the old
Kaggle + FLUX.1-schnell pipeline.

- No GPU, no notebook orchestration — direct HTTPS calls from the app.
- Resumable: an image is only (re)generated if its file doesn't already
  exist, so failed or interrupted runs can simply be re-triggered.
- Adaptive rate limiting: paces requests based on live success/failure
  feedback instead of a fixed guess, so it uses as much of the real
  per-minute headroom as it safely can.
- A failing scene never blocks the ones after it — it's parked and
  automatically retried in later rounds, landing under the same
  scene_id filename so it slots into the correct place in the video.
- Tries the best available free model first and automatically falls
  back down the candidate list if one isn't available on this API key.
"""

import os
import re
import time
import json
import hashlib
import tempfile
import shutil

RUNS_DIR = os.path.join(tempfile.gettempdir(), "pipeline_runs")

# Best-quality-first. The run tries these in order and sticks with the
# first one that actually works for this API key, falling back further
# down the list only if a model turns out to be unavailable/deprecated.
# Add new model IDs to the TOP of this list as Google ships better free
# image models — no other code changes needed for the pipeline to adopt them.
MODEL_CANDIDATES = [
    "gemini-3.1-flash-image-preview",  # "Nano Banana 2" — newer/better quality where available
    "gemini-2.5-flash-image",          # "Nano Banana" — proven free-tier workhorse
    "gemini-2.0-flash-exp",            # older fallback, last resort
]

# How many extra automatic rounds to retry scenes that failed in the main
# pass, after finishing every other scene once.
RETRY_ROUNDS = 3

# If a model returns rate-limit-class errors this many scenes in a row with
# no successes in between, treat it as an exhausted daily quota rather than
# a per-minute limit and switch models — otherwise the adaptive limiter just
# backs off forever and the whole run stalls at zero throughput.
RATE_LIMIT_STRIKES_BEFORE_SWITCH = 6

# Prepended to every scene prompt to keep the established visual style
# consistent across the whole batch. Edit this if the house style changes.
STYLE_PREFIX = (
    "Simple black and white stick-figure illustration, minimal line art "
    "style, plain background. Scene: "
)


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

    # A malformed/blank scene_id (possible after manual CSV edits) must
    # never silently drop a row from generation — coerce with a positional
    # fallback so every row still gets an image and a spot in the video.
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
    moment a rate-limit response is seen. Avoids both guessing too low
    (sitting idle when more headroom exists) and too high (burning through
    the quota and stalling on 429s).
    """

    def __init__(self, floor_rpm=6, ceiling_rpm=20, start_rpm=10, successes_per_speedup=4):
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
        self._consecutive_rate_limit_failures = 0

    def current(self):
        return self.candidates[self.index]

    def advance(self):
        if self.index < len(self.candidates) - 1:
            self.index += 1
            self._consecutive_rate_limit_failures = 0
            return True
        return False

    def note_success(self):
        self._consecutive_rate_limit_failures = 0

    def note_rate_limited_failure(self):
        """
        A single 429 usually just means 'slow down' and the adaptive
        limiter already handles that. But if a model keeps coming back
        rate-limited scene after scene with no successes in between, that's
        much more likely a fully exhausted daily quota than a per-minute
        limit — in which case backing off further just wastes hours doing
        nothing. After enough consecutive strikes, treat it like the model
        is unavailable and move to the next candidate.
        """
        self._consecutive_rate_limit_failures += 1
        if self._consecutive_rate_limit_failures >= RATE_LIMIT_STRIKES_BEFORE_SWITCH:
            self.advance()


class AuthenticationError(RuntimeError):
    """Raised immediately, without retrying or switching models — a bad
    API key won't fix itself by waiting or trying a different model."""
    pass


def _classify_error(err_text):
    lowered = err_text.lower()
    if any(tok in lowered for tok in
           ["api_key_invalid", "invalid api key", "unauthenticated", "401",
            "api key not valid"]):
        return "auth_error"
    # A quota error reporting "limit: 0" means this model has NO free-tier
    # allocation at all on this key (often a preview model that's still
    # allowlist-only despite being documented as available) — not a busy
    # model that'll free up if we wait. No amount of backoff fixes a hard
    # zero, so treat it the same as model-not-found: switch immediately.
    if re.search(r"limit['\"]?\s*[:=]\s*0\b", lowered):
        return "model_unavailable"
    if "429" in err_text or "resource_exhausted" in lowered or "quota" in lowered or "rate" in lowered:
        return "rate_limit"
    if any(tok in lowered for tok in
           ["safety", "blocked", "prohibited_content", "recitation"]):
        return "safety_blocked"
    if any(tok in lowered for tok in
           ["404", "not_found", "not found", "is not supported",
            "invalid model", "does not exist", "permission_denied", "unsupported"]):
        return "model_unavailable"
    return "other"


def _attempt_generate(client, types_module, model_name, prompt_text, out_path):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt_text,
        config=types_module.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # Defensive parsing: a safety-filtered or empty response can leave
    # candidates/content/parts as None rather than an empty list, which
    # would otherwise surface as a raw, unclassifiable AttributeError.
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("blocked: response returned no candidates (likely safety filter)")

    finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            image = part.as_image()
            image.save(out_path)
            return True

    if finish_reason and finish_reason.upper() not in ("STOP", ""):
        raise RuntimeError(f"blocked: finish_reason={finish_reason}")
    return False


def _generate_with_fallback(client, types_module, model_state, prompt_text, out_path, rate_limiter):
    """
    Single "best effort" attempt at one image: tries the current model
    (with one same-model retry for rate-limit/transient hiccups), and
    switches to the next candidate model when the failure looks like a
    model-availability problem OR a safety block (a different model may
    simply not filter that prompt). Raises on total failure — the caller
    decides what happens next (park it for a later retry round).

    An authentication error is different: it raises immediately, with no
    retry and no model switching, since a bad key won't start working by
    waiting or trying another model name.
    """
    last_err = None

    for _ in range(len(model_state.candidates)):
        model_name = model_state.current()

        for local_attempt in range(2):
            try:
                if _attempt_generate(client, types_module, model_name, prompt_text, out_path):
                    rate_limiter.record_success()
                    return
                last_err = "No image data returned in response."
                break
            except Exception as e:
                err_text = str(e)
                kind = _classify_error(err_text)
                last_err = err_text

                if kind == "auth_error":
                    raise AuthenticationError(
                        f"Gemini API key was rejected: {err_text}"
                    )
                elif kind in ("model_unavailable", "safety_blocked"):
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

        kind = _classify_error(last_err or "")
        if kind in ("model_unavailable", "safety_blocked"):
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
    Generates one image per manifest row via the best available Gemini
    image model.

    progress_callback(done_images, total_images, message)

    Returns (images_dir, zip_path, failed_scene_ids). failed_scene_ids is
    only non-empty if a scene still failed after every retry round —
    re-calling this function later will retry just those.
    """
    from google import genai
    from google.genai import types

    if not api_key:
        raise ValueError("Missing Gemini API key.")

    df = _load_manifest_df(manifest_path)
    images_dir = get_images_dir(manifest_path)
    client = genai.Client(api_key=api_key)

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
        progress_callback(done_images, total_images, f"Starting with image model: {model_state.current()}")

    def _run_one(scene_id, prompt, out_path):
        nonlocal done_images, last_model_reported
        full_prompt = STYLE_PREFIX + prompt
        rate_limiter.wait()
        try:
            _generate_with_fallback(client, types, model_state, full_prompt, out_path, rate_limiter)
            done_images += 1
            model_state.note_success()
            if progress_callback:
                progress_callback(done_images, total_images,
                                   f"Scene {scene_id} done (~{rate_limiter.current_rpm} req/min).")
            if model_state.current() != last_model_reported:
                last_model_reported = model_state.current()
                if progress_callback:
                    progress_callback(done_images, total_images,
                                       f"Switched to fallback image model: {last_model_reported}")
            return True
        except AuthenticationError:
            # Not worth burning through the rest of the batch on a key that
            # will never succeed — let this stop the whole run immediately.
            raise
        except Exception as e:
            if _classify_error(str(e)) == "rate_limit":
                model_state.note_rate_limited_failure()
            if progress_callback:
                progress_callback(done_images, total_images, f"Scene {scene_id} failed, parked for retry: {e}")
            return False

    # Main pass — one shot per scene, in order. A failure is parked, not
    # retried in place, so it never holds up the scenes after it.
    failed = []
    for scene_id, prompt, out_path in pending:
        if not _run_one(scene_id, prompt, out_path):
            failed.append((scene_id, prompt, out_path))

    # Automatic retry rounds over whatever's still missing. Each success
    # writes straight to its scene_id filename, so it lands in the correct
    # sequence/timestamp for the video step without any manual re-run.
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

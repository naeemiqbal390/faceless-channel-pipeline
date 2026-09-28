"""
cloudflare_runner.py
Image generation via Cloudflare Workers AI — replaces pollinations_runner.py
entirely, as the deliberate quality-over-Pollinations trade-off.

WHY THE SWITCH: Pollinations' free "flux" is the schnell variant — distilled
to render in 4 steps, built for speed over fidelity. Cloudflare's
stable-diffusion-xl-base-1.0 runs a genuinely full 20-step diffusion
process — structurally a different (higher-detail, better-delineated)
trade-off, at the cost of being slower and metered by a daily quota rather
than Pollinations' confusing one-time Pollen credit.

THE QUOTA: 10,000 free Neurons/day, no credit card, resetting at 00:00 UTC.
Full-step SDXL costs meaningfully more Neurons per image than a 4-step
model — expect on the rough order of 40-50 SDXL images per day before the
free allocation runs out for that day. This is a genuine constraint, not a
bug: once exhausted, remaining scenes fail and simply stay parked. Thanks
to the B2 persistence layer, there's nothing to do about that except come
back after the next UTC reset and click Resume — generation continues
exactly where it stopped, no re-work, no manual bookkeeping.

MODEL HIERARCHY (best first, automatic fallback down the list):
  1. @cf/leonardo/phoenix-1.0      — Leonardo's own flagship model, now
                                      offered via Cloudflare; not available
                                      on every account, so this is a bonus
                                      if it works, not a guarantee.
  2. @cf/stabilityai/stable-diffusion-xl-base-1.0 — the confirmed, reliable,
                                      full-step-diffusion workhorse.
  3. @cf/black-forest-labs/flux-1-schnell — fast last resort, keeps a batch
                                      moving if SDXL itself is unavailable.

Same resumable / non-blocking-failure / automatic-retry-round design as the
Pollinations version it replaces: a failed scene is parked and retried
later without blocking the scenes after it, and successes always land
under their correct scene_id filename so they slot into the right place
in the video.
"""

import os
import time
import json
import random
import hashlib
import tempfile
import shutil

RUNS_DIR = os.path.join(tempfile.gettempdir(), "pipeline_runs")

API_BASE = "https://api.cloudflare.com/client/v4/accounts"

MODEL_CANDIDATES = [
    "@cf/leonardo/phoenix-1.0",
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "@cf/black-forest-labs/flux-1-schnell",
]

RETRY_ROUNDS = 3

# Cloudflare's documented rate limit for these models is generous (720/min)
# — the real constraint is the daily Neuron budget, not request pace, so
# pacing here just avoids hammering the API pointlessly, not rationing.
FLOOR_RPM = 20
CEILING_RPM = 60
START_RPM = 40

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
NUM_STEPS = 20  # SDXL's documented maximum — full diffusion, not distilled

MIN_VALID_IMAGE_BYTES = 2000

# Applied to every style, on top of that style's own specific negative
# prompt. Grouped by the actual failure categories seen in real output
# (including a fused/malformed-anatomy creature confirmed from a real
# generated video) — a short generic negative prompt is not enough to
# reliably suppress anatomical distortion; it needs to name the specific
# failure modes.
UNIVERSAL_NEGATIVE = (
    "two heads, multiple heads, extra heads, duplicate head, second head, "
    "conjoined, siamese twins, extra limbs, missing limbs, extra legs, "
    "extra arms, too many legs, too many arms, fused limbs, merged limbs, "
    "melted limbs, floating limbs, disconnected limbs, malformed hands, "
    "extra fingers, missing fingers, fused fingers, mutated hands, "
    "disfigured face, asymmetric face, distorted face, warped face, "
    "melted face, deformed body, mutated anatomy, bad anatomy, "
    "anatomically incorrect, disproportionate body, elongated body, "
    "warped proportions, extra body parts, malformed anatomy, "
    "cloned face, duplicate body parts, "
    "text, words, letters, numbers, typography, captions, subtitles, "
    "watermark, logo, signature, username, stamp, garbled text, "
    "illegible text, "
    "blurry, out of focus, low resolution, pixelated, jpeg artifacts, "
    "compression artifacts, noise, grain, low contrast, washed out, "
    "muddy colors, overexposed, underexposed, flat lighting, no depth, "
    "smudged, smeared, muddled details, undefined edges, "
    "cropped, out of frame, cut off, duplicate, cloned, tiling, collage, "
    "multiple panels, split screen, extra background elements, "
    "worst quality, low quality, poorly drawn, ugly"
)

STYLE_PRESETS = {
    "Stick Figure": {
        "prompt": (
            "Extremely simple 2D vector clip-art icon. Pure white flat "
            "background. A minimalist stick-figure character made of thin, "
            "uniform, solid black ink outlines only: a plain circle for the "
            "head, straight simple lines for the body and limbs. Flat "
            "graphic design, like a pictogram or clipart icon. Scene: "
        ),
        "negative": (
            "photorealistic, 3d render, realistic skin, realistic anatomy, "
            "shading, gradient, painting, watercolor, airbrush, soft focus, "
            "blurry, textured, abstract, amorphous blob, noise, complex "
            "detail, photography, cinematic lighting, film grain"
        ),
    },
    "Anime / Hand-Painted (Ghibli-inspired)": {
        "prompt": (
            "Hand-painted 2D anime background art, soft watercolor palette, "
            "whimsical storybook atmosphere, warm natural lighting. Scene: "
        ),
        "negative": "photorealistic, 3d render, photography, blurry, low detail, abstract",
    },
    "1980s Retro Anime": {
        "prompt": (
            "1980s retro anime style, grainy film texture, bold cel-shaded "
            "colors, vintage VHS aesthetic. Scene: "
        ),
        "negative": "modern digital art, 3d render, photorealistic, blurry, low detail",
    },
    "Watercolor": {
        "prompt": (
            "Soft watercolor painting, gentle visible brush strokes, muted "
            "pastel color palette. Scene: "
        ),
        "negative": "photorealistic, 3d render, digital vector art, hard edges, blurry",
    },
    "Comic Book": {
        "prompt": (
            "Bold comic book illustration, heavy ink outlines, halftone "
            "shading, vibrant saturated colors. Scene: "
        ),
        "negative": "photorealistic, 3d render, watercolor, soft focus, blurry, muted colors",
    },
    "Photorealistic": {
        "prompt": (
            "Photorealistic, cinematic lighting, high detail, shot on 35mm "
            "film. Scene: "
        ),
        "negative": "cartoon, illustration, line art, painting, low detail, blurry, abstract",
    },
}
DEFAULT_STYLE = "Stick Figure"

ANATOMY_POSITIVE = (
    "Correct, coherent anatomy: exactly one head, exactly one face, "
    "the correct number of limbs, symmetrical features, natural "
    "proportions. "
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
    def __init__(self, floor_rpm=FLOOR_RPM, ceiling_rpm=CEILING_RPM, start_rpm=START_RPM,
                 successes_per_speedup=4):
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
    if any(tok in lowered for tok in
           ["authentication", "invalid token", "unauthorized", "10000", "401"]):
        return "auth_error"
    if any(tok in lowered for tok in
           ["neuron", "quota", "daily limit", "budget"]):
        return "quota_exhausted"  # account-wide — no point switching models, just park it
    if "429" in err_text or "too many requests" in lowered or "rate limit" in lowered:
        return "rate_limit"
    if any(tok in lowered for tok in
           ["3003", "3004", "not found", "no such model", "unsupported", "not available", "404"]):
        return "model_unavailable"
    return "other"


class AuthenticationError(RuntimeError):
    pass


def _attempt_generate(account_id, api_token, model_name, prompt_text, negative_prompt, out_path):
    import requests

    url = f"{API_BASE}/{account_id}/ai/run/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "prompt": prompt_text,
        "negative_prompt": negative_prompt,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "num_steps": NUM_STEPS,
        "guidance": 7.5,
        "seed": random.randint(1, 2_147_483_647),
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        if len(response.content) < MIN_VALID_IMAGE_BYTES:
            raise RuntimeError(f"response too small to be a real image ({len(response.content)} bytes)")
        with open(out_path, "wb") as f:
            f.write(response.content)
        return True

    # Not an image — this is Cloudflare's standard {"success": false, ...}
    # error envelope, or an unexpected response shape.
    raise RuntimeError(f"{response.status_code}: {response.text[:400]}")


def _generate_with_fallback(account_id, api_token, model_state, prompt_text, negative_prompt,
                             out_path, rate_limiter):
    last_err = None

    for _ in range(len(model_state.candidates)):
        model_name = model_state.current()

        for local_attempt in range(2):
            try:
                _attempt_generate(account_id, api_token, model_name, prompt_text, negative_prompt, out_path)
                rate_limiter.record_success()
                return
            except Exception as e:
                err_text = str(e)
                kind = _classify_error(err_text)
                last_err = err_text

                if kind == "auth_error":
                    raise AuthenticationError(f"Cloudflare credentials were rejected: {err_text}")
                elif kind == "quota_exhausted":
                    # Account-wide daily budget is gone — retrying or
                    # switching models won't help until the next UTC reset.
                    raise RuntimeError(f"Daily Neuron quota exhausted: {err_text}")
                elif kind == "model_unavailable":
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

def run_image_generation(manifest_path, api_key, progress_callback=None, style=DEFAULT_STYLE,
                          on_image_saved=None, account_id=None):
    """
    Generates one image per manifest row via Cloudflare Workers AI.

    'api_key' is the Cloudflare API token (required — Workers AI has no
    anonymous tier). 'account_id' is your Cloudflare account ID (required).

    progress_callback(done_images, total_images, message)
    on_image_saved(scene_id, local_path) — called right after each image is
    written, before moving to the next scene.

    Returns (images_dir, zip_path, failed_scene_ids).
    """
    if not api_key:
        raise ValueError("Missing Cloudflare API token.")
    if not account_id:
        raise ValueError("Missing Cloudflare account ID.")

    style_config = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])
    style_prefix = style_config["prompt"] + ANATOMY_POSITIVE
    negative_prompt = style_config.get("negative", "") + ", " + UNIVERSAL_NEGATIVE

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
    quota_exhausted = False

    if progress_callback and pending:
        progress_callback(done_images, total_images, f"Starting with model: {model_state.current()}")

    def _run_one(scene_id, prompt, out_path):
        nonlocal done_images, last_model_reported, quota_exhausted
        full_prompt = style_prefix + prompt
        rate_limiter.wait()
        try:
            _generate_with_fallback(account_id, api_key, model_state, full_prompt, negative_prompt,
                                     out_path, rate_limiter)
            done_images += 1
            if on_image_saved:
                try:
                    on_image_saved(scene_id, out_path)
                except Exception:
                    pass
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
            raise
        except Exception as e:
            if "daily neuron quota exhausted" in str(e).lower():
                quota_exhausted = True
            if progress_callback:
                progress_callback(done_images, total_images, f"Scene {scene_id} failed, parked for retry: {e}")
            return False

    failed = []
    for scene_id, prompt, out_path in pending:
        if quota_exhausted:
            # No point burning through remaining scenes one-by-one once the
            # account-wide daily budget is confirmed gone — park the rest
            # immediately. They'll resume cleanly after the next UTC reset.
            failed.append((scene_id, prompt, out_path))
            continue
        if not _run_one(scene_id, prompt, out_path):
            failed.append((scene_id, prompt, out_path))

    if quota_exhausted and progress_callback:
        progress_callback(done_images, total_images,
                           "Daily Cloudflare Neuron quota exhausted — remaining scenes parked. "
                           "Resets at 00:00 UTC; come back and click Resume/Generate to continue.")

    round_num = 1
    while failed and round_num <= RETRY_ROUNDS and not quota_exhausted:
        if progress_callback:
            progress_callback(done_images, total_images,
                               f"Retry round {round_num}/{RETRY_ROUNDS} — {len(failed)} scene(s) remaining.")
        still_failed = []
        for scene_id, prompt, out_path in failed:
            if quota_exhausted:
                still_failed.append((scene_id, prompt, out_path))
                continue
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

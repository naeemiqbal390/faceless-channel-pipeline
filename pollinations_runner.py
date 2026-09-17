"""
pollinations_runner.py
Image generation via Pollinations.ai.

TWO SEPARATE AUTH SYSTEMS, AUTO-DETECTED BY KEY FORMAT:
- No key, or a plain legacy token: uses the older, genuinely free, no-signup
  endpoint at image.pollinations.ai. Anonymous is throttled (~1 req/15s) and
  always watermarked; the legacy "token" registration system only ever
  affected this endpoint.
- A key starting with "sk_" or "pk_": this is a credential for Pollinations'
  newer metered gateway at gen.pollinations.ai. It unlocks all premium
  models, no watermark, and much higher throughput — but EVERY image,
  including flux, spends Pollen credit from that key's balance. It's cheap,
  but no longer literally free the way anonymous access is. Check your
  balance at enter.pollinations.ai.
- MODEL HIERARCHY: tries the best available model first, and automatically
  falls back down the list the moment a model is unavailable or (on the
  metered gateway) out of credit.
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

LEGACY_BASE_URL = "https://image.pollinations.ai/prompt/"
GATEWAY_BASE_URL = "https://gen.pollinations.ai/image/"


def _is_gateway_key(token):
    return bool(token) and token.startswith(("sk_", "pk_"))


# Best-quality-first. Premium models are only reachable via the metered
# gen.pollinations.ai gateway (an sk_/pk_ key); "flux"/"turbo" work on
# both, free and unlimited on the legacy anonymous endpoint. The run tries
# top-down and permanently drops to the next candidate the moment a model
# reports itself unavailable or out of credit — add new premium model
# names here as Pollinations ships them, no other code changes needed.
MODEL_CANDIDATES_GATEWAY = ["nanobanana", "seedream", "gptimage", "flux"]
MODEL_CANDIDATES_LEGACY = ["flux"]  # turbo deliberately excluded — lower quality tier; retry rounds keep trying flux instead of silently downgrading

RETRY_ROUNDS = 3

# Anonymous legacy use is documented at roughly one request per 15 seconds.
# The metered gateway (an actual account with credit) tolerates much higher
# throughput.
LEGACY_PACING = dict(floor_rpm=2, ceiling_rpm=5, start_rpm=4)
GATEWAY_PACING = dict(floor_rpm=4, ceiling_rpm=15, start_rpm=8)

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# A real image is never this small — anything under this is almost
# certainly an HTML error page or truncated response, not a photo.
MIN_VALID_IMAGE_BYTES = 2000

# Models whose "quality" parameter Pollinations documents as actually
# honored (others are said to ignore it, so there's no harm sending it
# broadly, but this is what it's known to matter for).
QUALITY_AWARE_MODELS = {"gptimage", "gptimage-large", "gpt-image-2"}

# Applied to every style, on top of that style's own specific negative
# prompt. Diffusion models are notoriously bad at rendering legible text —
# if your scene prompts ever ask for on-screen captions/labels, this won't
# make that work, it'll just stop a failed text attempt from wrecking the
# rest of the composition. The real fix for on-screen text is to burn it in
# with ffmpeg after generation, not ask the image model to paint the words.
# Applied to every style, on top of that style's own specific negative
# prompt. Grouped by the actual failure categories seen in real output
# (including a fused/malformed-anatomy creature confirmed from a real
# generated video) — a short generic negative prompt is not enough to
# reliably suppress anatomical distortion; it needs to name the specific
# failure modes. Diffusion models are also notoriously bad at rendering
# legible text — if scene prompts ever ask for on-screen captions/labels,
# no negative prompt fixes that; the real fix is burning text in with
# ffmpeg after generation, not asking the image model to paint words.
UNIVERSAL_NEGATIVE = (
    # Anatomy / malformation — the category that produced the fused,
    # multi-limbed creature seen in testing
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
    # Text — models can't render legible text reliably
    "text, words, letters, numbers, typography, captions, subtitles, "
    "watermark, logo, signature, username, stamp, garbled text, "
    "illegible text, "
    # Rendering / technical quality
    "blurry, out of focus, low resolution, pixelated, jpeg artifacts, "
    "compression artifacts, noise, grain, low contrast, washed out, "
    "muddy colors, overexposed, underexposed, flat lighting, no depth, "
    "smudged, smeared, muddled details, undefined edges, "
    # Composition defects
    "cropped, out of frame, cut off, duplicate, cloned, tiling, collage, "
    "multiple panels, split screen, extra background elements, "
    "worst quality, low quality, poorly drawn, ugly"
)

# Visual style presets — prepended to every scene prompt, paired with a
# negative_prompt that actively steers the model AWAY from the qualities
# that tend to sabotage that particular style (e.g. diffusion models often
# drift toward soft painterly renders even when asked for plain line art,
# unless something actively pushes back against that). "Style" here means
# a general aesthetic descriptor (palette, line work, era), not a request
# to reproduce any specific studio's copyrighted characters or film stills.
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

# Appended after every style's own prompt, right before the scene content.
# Pairing an explicit positive instruction with the negative prompt is
# meaningfully more effective than negation alone for anatomy specifically —
# models respond better to being told what correct looks like, not just
# what to avoid.
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


def _attempt_generate(model_name, prompt_text, out_path, token=None, negative_prompt=None):
    import requests

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
    if negative_prompt:
        params["negative_prompt"] = negative_prompt

    headers = {}
    if _is_gateway_key(token):
        url = GATEWAY_BASE_URL + quote(prompt_text)
        headers["Authorization"] = f"Bearer {token}"
    else:
        url = LEGACY_BASE_URL + quote(prompt_text)
        if token:
            params["token"] = token  # legacy-style token, not an sk_/pk_ key

    response = requests.get(url, params=params, headers=headers, timeout=90)

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


def _generate_with_fallback(model_state, prompt_text, out_path, rate_limiter, token, negative_prompt=None):
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
                _attempt_generate(model_name, prompt_text, out_path, token, negative_prompt)
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

def run_image_generation(manifest_path, api_key, progress_callback=None, style=DEFAULT_STYLE, on_image_saved=None):
    """
    Generates one image per manifest row via Pollinations.ai.

    'api_key' is an OPTIONAL Pollinations account token — pass "" or None to
    use anonymously (free models only, slower pacing). With a token, premium
    models are tried first and pacing assumes the faster registered tier.

    'style' selects a STYLE_PRESETS key to prepend to every scene prompt.

    'on_image_saved(scene_id, local_path)' — optional hook called right
    after each image is successfully written, before moving to the next
    scene. Used to persist images to remote storage incrementally rather
    than waiting for the whole batch to finish.

    progress_callback(done_images, total_images, message)

    Returns (images_dir, zip_path, failed_scene_ids).
    """
    token = api_key or None
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

    use_gateway = _is_gateway_key(token)
    candidates = MODEL_CANDIDATES_GATEWAY if use_gateway else MODEL_CANDIDATES_LEGACY
    pacing = GATEWAY_PACING if use_gateway else LEGACY_PACING
    model_state = ModelState(list(candidates))
    rate_limiter = AdaptiveRateLimiter(**pacing)
    last_model_reported = model_state.current()

    if progress_callback and pending:
        # Unambiguous either way — a masked preview of the actual key
        # received (never the full value) proves definitively whether the
        # app is even seeing a token, rather than inferring it indirectly.
        if token:
            masked = token[:6] + "…" + token[-4:] if len(token) > 12 else token[:3] + "…"
        else:
            masked = "(none)"
        if use_gateway:
            mode_note = f"using metered gateway (gen.pollinations.ai), key={masked} — spends Pollen credit per image"
        elif token:
            mode_note = f"using legacy token (image.pollinations.ai), key={masked} — free, but not sk_/pk_ format"
        else:
            mode_note = "no key received — anonymous legacy endpoint, free but watermarked and rate-limited"
        progress_callback(done_images, total_images,
                           f"Starting with model: {model_state.current()} ({mode_note}, style: {style}).")

    def _run_one(scene_id, prompt, out_path):
        nonlocal done_images, last_model_reported
        full_prompt = style_prefix + prompt
        rate_limiter.wait()
        try:
            _generate_with_fallback(model_state, full_prompt, out_path, rate_limiter, token, negative_prompt)
            done_images += 1
            if on_image_saved:
                try:
                    on_image_saved(scene_id, out_path)
                except Exception:
                    pass  # persistence hook failing must never abort generation itself
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

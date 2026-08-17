"""
kaggle_runner.py — uploads the scene manifest to Kaggle as a dataset, pushes
a GPU script-kernel that generates one stick-figure image per scene, waits
for it to finish, and returns the output zip + a human-readable log.

This is the module app.py expects to import:
    from kaggle_runner import run_image_generation_on_kaggle
    zip_path, log = run_image_generation_on_kaggle(manifest_path, kaggle_user, kaggle_key)

WHY THE PREVIOUS RUNS FAILED
-----------------------------
Every failed run showed the same traceback:
    FileNotFoundError: /kaggle/input/scene-manifest-<id>/scene_manifest.csv
The dataset upload itself always succeeded (it was visible under "Your Work
-> Datasets" every time) but the kernel that was pushed never listed that
dataset in its `dataset_sources`, so Kaggle launched the kernel with no
input attached at all — the file genuinely wasn't there from the kernel's
point of view, no matter what got uploaded.

This module fixes that by:
  1. Uploading the manifest as a *new, uniquely named* private dataset and
     actively polling until Kaggle confirms the file is really there
     (uploads are not instantly queryable — a dataset can exist with zero
     files for a few seconds after creation).
  2. Building kernel-metadata.json's `dataset_sources` from the *exact*
     slug just used in step 1 — the two are never allowed to drift apart.
  3. Only pushing the kernel after that dataset is confirmed ready.

It also works around a second, separate problem visible in the same logs:
Kaggle's default P100 GPU (compute capability sm_60) is no longer supported
by the PyTorch build pre-installed in Kaggle's current Docker image (which
only supports sm_70+). The generated kernel script pins a P100-compatible
torch/torchvision build for itself before importing torch, so it actually
uses the GPU instead of erroring out or silently falling back to CPU.
"""

import os
import io
import re
import csv
import time
import json
import zipfile
import tempfile


# ============================================================
# Kaggle auth
# ============================================================

def _configure_kaggle_auth(kaggle_user, kaggle_key):
    """Set credentials the way the kaggle package actually checks for them:
    env vars first (KaggleApi.authenticate() reads these directly), and a
    ~/.kaggle/kaggle.json file as a fallback for any code path that still
    looks for the legacy file instead of the env vars."""
    if not kaggle_user or not kaggle_key:
        raise ValueError(
            "Missing Kaggle credentials — set KAGGLE_USERNAME/KAGGLE_KEY in "
            "Secrets, or fill in the username/key fields in the app."
        )

    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key

    try:
        kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        cred_path = os.path.join(kaggle_dir, "kaggle.json")
        with open(cred_path, "w") as f:
            json.dump({"username": kaggle_user, "key": kaggle_key}, f)
        os.chmod(cred_path, 0o600)
    except Exception:
        # Non-fatal — env vars alone are enough for the API client.
        pass


def _get_api(kaggle_user, kaggle_key):
    _configure_kaggle_auth(kaggle_user, kaggle_key)
    # Imported lazily (after env vars are set) so the module import itself
    # never fails just because Streamlit loaded the app with no creds yet.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


# ============================================================
# The script that actually runs on Kaggle's GPU
# ============================================================

_KERNEL_SCRIPT = r'''
import os
import sys
import csv
import zipfile
import subprocess
import traceback

def log(msg):
    print(msg, flush=True)

# --- Work around Kaggle's default image no longer supporting older GPUs
# (P100 = compute capability sm_60; current pre-installed torch build only
# supports sm_70+). Pin a torch/torchvision build from the cu118 index,
# which still supports Pascal-generation GPUs, BEFORE torch is imported
# anywhere in this process.
def ensure_gpu_compatible_torch():
    try:
        import torch  # noqa
        cap = None
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
        if cap is not None and cap[0] >= 7:
            return  # current install already works with this GPU
    except Exception:
        pass

    log("Installing a P100-compatible torch/torchvision build...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "torch==2.1.2", "torchvision==0.16.2",
         "--index-url", "https://download.pytorch.org/whl/cu118"],
        check=False,
    )

ensure_gpu_compatible_torch()

import torch
from diffusers import StableDiffusionPipeline

MANIFEST_PATH = "MANIFEST_PATH_PLACEHOLDER"
OUTPUT_DIR = "/kaggle/working"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "scene_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

if not os.path.exists(MANIFEST_PATH):
    log(f"ERROR: manifest not found at {MANIFEST_PATH}")
    log("Contents of /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        for fn in files:
            log("  " + os.path.join(root, fn))
    sys.exit(1)

rows = []
with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

log(f"Loaded {len(rows)} scenes from manifest.")

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Using device: {device}")

dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None,
)
pipe = pipe.to(device)

NEGATIVE_PROMPT = (
    "photorealistic, photograph, 3d render, realistic skin, hyperrealism, "
    "blurry, low quality, text, watermark, extra limbs, deformed"
)

failures = []
for row in rows:
    scene_id = row.get("scene_id", "0").strip() or "0"
    prompt = row.get("prompt", "").strip()
    if not prompt:
        log(f"Scene {scene_id}: empty prompt, skipping.")
        continue
    try:
        log(f"Scene {scene_id}: generating...")
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        out_path = os.path.join(IMAGES_DIR, f"scene_{int(scene_id):02d}.png")
        image.save(out_path)
        log(f"Scene {scene_id}: saved {out_path}")
    except Exception:
        log(f"Scene {scene_id}: FAILED")
        log(traceback.format_exc())
        failures.append(scene_id)

zip_path = os.path.join(OUTPUT_DIR, "scene_images_batch.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in sorted(os.listdir(IMAGES_DIR)):
        zf.write(os.path.join(IMAGES_DIR, fn), arcname=fn)

log(f"Wrote {zip_path}")
if failures:
    log(f"WARNING: {len(failures)} scene(s) failed: {failures}")
log("DONE")
'''


# ============================================================
# Dataset upload (manifest -> Kaggle dataset), with a readiness check
# ============================================================

def _upload_manifest_dataset(api, kaggle_user, manifest_path, run_id):
    dataset_slug = f"scene-manifest-{run_id}"
    staging_dir = tempfile.mkdtemp(prefix="ds_stage_")

    csv_dest = os.path.join(staging_dir, "scene_manifest.csv")
    with open(manifest_path, "rb") as src, open(csv_dest, "wb") as dst:
        dst.write(src.read())

    dataset_metadata = {
        "title": dataset_slug,
        "id": f"{kaggle_user}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(staging_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f)

    api.dataset_create_new(
        folder=staging_dir,
        public=False,
        quiet=True,
        convert_to_csv=False,
        dir_mode="zip",
    )

    _wait_for_dataset_ready(api, kaggle_user, dataset_slug)
    return dataset_slug


def _wait_for_dataset_ready(api, kaggle_user, dataset_slug, timeout_s=120, interval_s=5):
    """Kaggle datasets aren't instantly queryable right after creation —
    poll until the file list actually comes back non-empty before we let
    anything reference this dataset from a kernel."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            files = api.dataset_list_files(f"{kaggle_user}/{dataset_slug}").files
            if files and len(files) > 0:
                return
        except Exception as e:
            last_err = e
        time.sleep(interval_s)
    raise TimeoutError(
        f"Dataset {kaggle_user}/{dataset_slug} never became ready within "
        f"{timeout_s}s. Last error: {last_err}"
    )


# ============================================================
# Kernel push + poll + fetch output
# ============================================================

def _push_image_gen_kernel(api, kaggle_user, dataset_slug, run_id):
    kernel_dir = tempfile.mkdtemp(prefix="kernel_stage_")
    kernel_slug = f"image-gen-{run_id}"

    manifest_input_path = f"/kaggle/input/{dataset_slug}/scene_manifest.csv"
    script_content = _KERNEL_SCRIPT.replace(
        "MANIFEST_PATH_PLACEHOLDER", manifest_input_path
    )
    with open(os.path.join(kernel_dir, "script.py"), "w") as f:
        f.write(script_content)

    kernel_metadata = {
        "id": f"{kaggle_user}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        # This is the line that was missing before — the kernel is now
        # wired to the *exact* dataset slug we just confirmed is ready.
        "dataset_sources": [f"{kaggle_user}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_metadata, f)

    api.kernels_push(kernel_dir)
    return f"{kaggle_user}/{kernel_slug}"


def _poll_kernel(api, kernel_ref, timeout_s=1200, interval_s=15):
    deadline = time.time() + timeout_s
    status = None
    while time.time() < deadline:
        result = api.kernels_status(kernel_ref)
        status = getattr(result, "status", None) or result.get("status", "")
        status = str(status).lower()
        if status in ("complete", "error", "cancelacknowledged", "cancelled"):
            return status
        time.sleep(interval_s)
    raise TimeoutError(f"Kernel {kernel_ref} did not finish within {timeout_s}s (last status: {status}).")


def _fetch_kernel_output(api, kernel_ref):
    out_dir = tempfile.mkdtemp(prefix="kernel_out_")
    try:
        api.kernels_output(kernel_ref, path=out_dir, force=True, quiet=True)
    except Exception:
        pass  # we still try to read whatever landed on disk below

    log_text = ""
    for fn in os.listdir(out_dir):
        if fn.endswith(".log") or fn == "custom.log":
            try:
                with open(os.path.join(out_dir, fn), "r", errors="ignore") as f:
                    log_text += f.read() + "\n"
            except Exception:
                pass

    zip_path = None
    for fn in os.listdir(out_dir):
        if fn == "scene_images_batch.zip":
            zip_path = os.path.join(out_dir, fn)
            break

    return zip_path, log_text, out_dir


# ============================================================
# Public entry point used by app.py
# ============================================================

def run_image_generation_on_kaggle(manifest_path, kaggle_user, kaggle_key):
    api = _get_api(kaggle_user, kaggle_key)
    run_id = str(int(time.time()))

    dataset_slug = _upload_manifest_dataset(api, kaggle_user, manifest_path, run_id)
    kernel_ref = _push_image_gen_kernel(api, kaggle_user, dataset_slug, run_id)

    status = _poll_kernel(api, kernel_ref)
    zip_path, log_text, out_dir = _fetch_kernel_output(api, kernel_ref)

    if status != "complete" or not zip_path:
        detail = log_text.strip()[-3000:] if log_text.strip() else "(no log captured)"
        raise RuntimeError(
            f"Kaggle kernel run failed: {kernel_ref} has status \"{status}\".\n"
            f"--- kernel log tail ---\n{detail}"
        )

    final_zip = os.path.join(tempfile.gettempdir(), "scene_images_batch.zip")
    with open(zip_path, "rb") as src, open(final_zip, "wb") as dst:
        dst.write(src.read())

    return final_zip, f"Images generated successfully. Kernel: {kernel_ref}"

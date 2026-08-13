"""
kaggle_runner.py

Handles talking to Kaggle's API on the user's behalf: uploads the scene
manifest as a dataset, pushes a kernel (notebook) that generates images
from it, polls until the run finishes, and downloads the output zip.

This is the piece that makes "Generate images" in the app a single click
instead of a manual Kaggle website visit.

NOTE: this has not been run against a live Kaggle account from within this
build environment (network restrictions here prevent it) — it follows
Kaggle's documented API contract, but expect to debug real edge cases
(auth format, dataset slug rules, kernel status polling timing) once this
runs for real on a deployed Space.
"""

import os
import json
import time
import shutil
import tempfile
import subprocess


KERNEL_SCRIPT_TEMPLATE = '''
import pandas as pd
import torch
from diffusers import FluxPipeline
import os, shutil

df = pd.read_csv("/kaggle/input/{dataset_slug}/scene_manifest.csv")
OUTPUT_DIR = "/kaggle/working/scene_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.to("cuda")

def safe_time(t):
    return str(t).replace(":", "-")

for _, row in df.iterrows():
    scene_id = int(row["scene_id"])
    fname = f"scene_{{scene_id:03d}}_{{safe_time(row['start_time'])}}_{{safe_time(row['end_time'])}}.png"
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path):
        continue
    image = pipe(prompt=row["prompt"], width=1024, height=576,
                 num_inference_steps=4, guidance_scale=0.0).images[0]
    image.save(out_path)
    print(f"Saved {{fname}}")

shutil.make_archive("/kaggle/working/scene_images_batch", "zip", OUTPUT_DIR)
print("DONE")
'''


def _run_kaggle_cli(args, env):
    result = subprocess.run(["kaggle"] + args, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed: {result.stderr}")
    return result.stdout


def run_image_generation_on_kaggle(manifest_path, kaggle_username, kaggle_key, timeout_minutes=60):
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = kaggle_username
    env["KAGGLE_KEY"] = kaggle_key

    work_dir = tempfile.mkdtemp(prefix="kaggle_run_")
    dataset_slug = f"{kaggle_username}/scene-manifest-{int(time.time())}"

    # 1. Prepare and upload the manifest as a Kaggle dataset
    dataset_dir = os.path.join(work_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    shutil.copy(manifest_path, os.path.join(dataset_dir, "scene_manifest.csv"))

    metadata = {
        "title": f"scene-manifest-{int(time.time())}",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f)

    _run_kaggle_cli(["datasets", "create", "-p", dataset_dir, "-q"], env)

    # 2. Prepare and push the generation kernel
    kernel_dir = os.path.join(work_dir, "kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    slug_name = dataset_slug.split("/")[-1]
    script_content = KERNEL_SCRIPT_TEMPLATE.format(dataset_slug=slug_name)
    with open(os.path.join(kernel_dir, "generate.py"), "w") as f:
        f.write(script_content)

    kernel_slug = f"{kaggle_username}/image-gen-{int(time.time())}"
    kernel_metadata = {
        "id": kernel_slug,
        "title": f"image-gen-{int(time.time())}",
        "code_file": "generate.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_slug],
    }
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_metadata, f)

    _run_kaggle_cli(["kernels", "push", "-p", kernel_dir], env)

    # 3. Poll for completion
    deadline = time.time() + timeout_minutes * 60
    status = "unknown"
    while time.time() < deadline:
        time.sleep(30)
        status_output = _run_kaggle_cli(["kernels", "status", kernel_slug], env)
        if "complete" in status_output.lower():
            status = "complete"
            break
        if "error" in status_output.lower() or "failed" in status_output.lower():
            raise RuntimeError(f"Kaggle kernel run failed: {status_output}")

    if status != "complete":
        raise RuntimeError("Kaggle run timed out before completing — check the kernel manually on kaggle.com.")

    # 4. Download the output
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    _run_kaggle_cli(["kernels", "output", kernel_slug, "-p", output_dir], env)

    zip_path = os.path.join(output_dir, "scene_images_batch.zip")
    if not os.path.exists(zip_path):
        raise RuntimeError("Kernel finished but scene_images_batch.zip was not found in the output.")

    log = f"Images generated successfully via Kaggle kernel {kernel_slug}."
    return zip_path, log

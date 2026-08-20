"""
kaggle_runner.py

Handles talking to Kaggle's API on the user's behalf, split into CHUNKS so
the app can show real, exact progress and genuinely resume after an
interruption — rather than one opaque black-box run with no visibility.
"""

import os
import re
import json
import time
import shutil
import hashlib
import tempfile
import subprocess

import pandas as pd


RUNS_DIR = os.path.join(tempfile.gettempdir(), "pipeline_runs")


def load_and_fix_manifest(csv_path):
    """
    Reads the manifest CSV and guarantees 'scene_id' and 'prompt' columns 
    exist with zero whitespace, BOM, or casing bugs.
    """
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manifest file not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_path)

    # Clean whitespace, BOM characters, and lowercase headers
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '').str.lower()

    # Map header aliases to expected names
    rename_dict = {}
    for col in df.columns:
        if col in ['scene', 'scene id', 'scene_number', 'id', 'sn', 'unnamed: 0']:
            rename_dict[col] = 'scene_id'
        elif col in ['prompts', 'image_prompt', 'scene_prompt', 'description', 'text']:
            rename_dict[col] = 'prompt'

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    # Hard Fallback: Auto-inject scene_id if completely missing
    if 'scene_id' not in df.columns:
        df['scene_id'] = range(1, len(df) + 1)

    # Hard Fallback: Auto-inject default prompt if missing
    if 'prompt' not in df.columns:
        df['prompt'] = "stick figure drawing"

    # Ensure scene_id is integer-based
    df['scene_id'] = pd.to_numeric(df['scene_id'], errors='coerce').fillna(range(1, len(df) + 1)).astype(int)
    
    # Save clean copy over the file
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return df


KERNEL_SCRIPT_TEMPLATE = '''
import os, sys, subprocess, shutil, traceback

def log(msg):
    print(msg, flush=True)

log("Installing stable diffusion dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.27.2", "transformers", "accelerate"], check=True)

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline

try:
    manifest_path = "/kaggle/input/{dataset_slug}/scene_manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {{manifest_path}}")

    try:
        df = pd.read_csv(manifest_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(manifest_path)

    # Strip whitespace, remove BOM characters, and force lowercase column names
    df.columns = df.columns.astype(str).str.strip().str.replace('\\ufeff', '').str.lower()

    OUTPUT_DIR = "/kaggle/working/scene_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"Loading Stable Diffusion on {{device}}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    log("Model loaded successfully.")

    NEGATIVE_PROMPT = "photorealistic, photograph, 3d render, realistic skin, hyperrealism, blurry, low quality, text, watermark"

    for idx, row in df.iterrows():
        # Safely extract scene_id with fallback options
        scene_id_val = row.get("scene_id", row.get("scene", idx + 1))
        try:
            scene_id = int(scene_id_val)
        except (ValueError, TypeError):
            scene_id = idx + 1

        fname = f"scene_{scene_id:03d}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
            
        prompt_text = str(row.get("prompt", "stick figure drawing")) if pd.notna(row.get("prompt")) else "stick figure drawing"
        image = pipe(
            prompt=prompt_text,
            negative_prompt=NEGATIVE_PROMPT,
            width=512, height=512,
            num_inference_steps=20, guidance_scale=7.5,
        ).images[0]
        image.save(out_path)
        log(f"Saved {{fname}}")

    shutil.make_archive("/kaggle/working/scene_images_batch", "zip", OUTPUT_DIR)
    log("DONE")
except Exception as e:
    log("KERNEL_SCRIPT_ERROR: " + str(e))
    traceback.print_exc()
    raise
'''


def _run_kaggle_cli(args, env):
    result = subprocess.run(["kaggle"] + args, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed: {result.stderr}")
    return result.stdout


def compute_run_id(manifest_path):
    """A stable ID derived from the manifest's content, so resuming the
    same video's manifest always finds the same saved progress."""
    try:
        with open(manifest_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "default_run"


def _load_state(run_id, manifest_path=""):
    state_path = os.path.join(RUNS_DIR, run_id, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                if isinstance(state, dict):
                    return state
        except Exception:
            pass
    return {"completed_chunks": [], "manifest_path": manifest_path}


def _save_state(run_id, state):
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    try:
        with open(os.path.join(run_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def get_resume_status(manifest_path, chunk_size=25):
    """Check if there's existing progress for this exact manifest safely."""
    try:
        df = load_and_fix_manifest(manifest_path)
        run_id = compute_run_id(manifest_path)
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        state = _load_state(run_id, manifest_path=manifest_path)
        done = len(state.get("completed_chunks", []))
        
        return {
            "run_id": run_id,
            "manifest_path": state.get("manifest_path", manifest_path),
            "total_chunks": total_chunks,
            "done_chunks": done,
            "total_images": len(df),
            "has_progress": done > 0,
        }
    except Exception:
        return {
            "run_id": "",
            "manifest_path": manifest_path,
            "total_chunks": 0,
            "done_chunks": 0,
            "total_images": 0,
            "has_progress": False,
        }


def _wait_for_dataset_ready(dataset_slug, env, timeout_s=180, interval_s=10):
    deadline = time.time() + timeout_s
    check_dir = tempfile.mkdtemp(prefix="ds_ready_check_")
    last_err = ""
    while time.time() < deadline:
        try:
            for fn in os.listdir(check_dir):
                os.remove(os.path.join(check_dir, fn))
            _run_kaggle_cli(
                ["datasets", "download", "-d", dataset_slug, "-f", "scene_manifest.csv",
                 "-p", check_dir, "--force"],
                env,
            )
            plain = os.path.join(check_dir, "scene_manifest.csv")
            zipped = os.path.join(check_dir, "scene_manifest.csv.zip")
            if os.path.exists(plain) and os.path.getsize(plain) > 0:
                return
            if os.path.exists(zipped) and os.path.getsize(zipped) > 0:
                return
            last_err = f"download ran but no file appeared in {check_dir}"
        except Exception as e:
            last_err = str(e)
        time.sleep(interval_s)
    raise RuntimeError(
        f"Dataset {dataset_slug} never became downloadable within {timeout_s}s.\n"
        f"Last error: {last_err}"
    )


def _fetch_kernel_log_tail(kernel_slug, env, max_chars=3000):
    out_dir = tempfile.mkdtemp(prefix="kernel_fail_out_")
    try:
        _run_kaggle_cli(["kernels", "output", kernel_slug, "-p", out_dir], env)
    except Exception as e:
        return f"(could not download kernel output: {e})"

    collected = ""
    try:
        for fn in sorted(os.listdir(out_dir)):
            full = os.path.join(out_dir, fn)
            if not os.path.isfile(full):
                continue
            if fn.endswith(".log") or "log" in fn.lower():
                with open(full, "r", errors="ignore") as f:
                    collected += f"--- {fn} ---\n" + f.read() + "\n"
    except Exception:
        pass

    if not collected:
        return f"(no log file found in kernel output; files present: {os.listdir(out_dir) if os.path.exists(out_dir) else '[]'})"
    return collected[-max_chars:]


def _is_manifest_not_found_signature(log_tail):
    if not log_tail:
        return False
    return "FileNotFoundError" in log_tail and "scene_manifest.csv" in log_tail


def _attempt_chunk_once(chunk_df, kaggle_username, kaggle_key, chunk_index,
                        attempt_number, timeout_minutes, env):
    work_dir = tempfile.mkdtemp(prefix=f"kaggle_chunk_{chunk_index}_a{attempt_number}_")
    dataset_slug_name = f"chunk-{chunk_index}-{int(time.time())}"
    dataset_slug = f"{kaggle_username}/{dataset_slug_name}"

    dataset_dir = os.path.join(work_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    chunk_df.to_csv(os.path.join(dataset_dir, "scene_manifest.csv"), index=False, encoding='utf-8')
    with open(os.path.join(dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump({"title": dataset_slug_name, "id": dataset_slug, "licenses": [{"name": "CC0-1.0"}]}, f)
    _run_kaggle_cli(["datasets", "create", "-p", dataset_dir, "-q"], env)

    _wait_for_dataset_ready(dataset_slug, env)

    extra_buffer_s = 60 * attempt_number
    print(f"Chunk {chunk_index}, attempt {attempt_number}: dataset downloadable, "
          f"waiting {extra_buffer_s}s extra for kernel-attachment indexing...")
    time.sleep(extra_buffer_s)

    kernel_dir = os.path.join(work_dir, "kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    with open(os.path.join(kernel_dir, "generate.py"), "w") as f:
        f.write(KERNEL_SCRIPT_TEMPLATE.format(dataset_slug=dataset_slug_name))
    kernel_slug_name = f"image-gen-chunk-{chunk_index}-a{attempt_number}-{int(time.time())}"
    kernel_slug = f"{kaggle_username}/{kernel_slug_name}"
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump({
            "id": kernel_slug, "title": kernel_slug_name, "code_file": "generate.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset_slug],
        }, f)
    _run_kaggle_cli(["kernels", "push", "-p", kernel_dir], env)

    deadline = time.time() + timeout_minutes * 60
    status = "unknown"
    last_status_text = ""
    while time.time() < deadline:
        time.sleep(20)
        status_output = _run_kaggle_cli(["kernels", "status", kernel_slug], env)
        last_status_text = status_output.strip()
        lowered = last_status_text.lower()
        if re.search(r"\berror\b|\bfailed\b|\bcancelled\b", lowered):
            log_tail = _fetch_kernel_log_tail(kernel_slug, env)
            raise _ChunkAttemptError(
                f"Chunk {chunk_index} crashed (attempt {attempt_number}). Status: {last_status_text}\n"
                f"Kaggle page: https://www.kaggle.com/code/{kernel_slug}\n"
                f"--- log tail ---\n{log_tail}",
                log_tail=log_tail,
            )
        if re.search(r"\bcomplete\b", lowered) and "incomplete" not in lowered:
            status = "complete"
            break

    if status != "complete":
        log_tail = _fetch_kernel_log_tail(kernel_slug, env)
        raise _ChunkAttemptError(
            f"Chunk {chunk_index} did not finish in {timeout_minutes} min (attempt {attempt_number}). "
            f"Last status: '{last_status_text}'.\n"
            f"Kaggle page: https://www.kaggle.com/code/{kernel_slug}\n"
            f"--- log tail ---\n{log_tail}",
            log_tail=log_tail,
        )

    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    _run_kaggle_cli(["kernels", "output", kernel_slug, "-p", output_dir], env)

    zip_path = os.path.join(output_dir, "scene_images_batch.zip")
    if not os.path.exists(zip_path):
        raise _ChunkAttemptError(
            f"Chunk {chunk_index} finished but produced no output zip (attempt {attempt_number}).",
            log_tail="",
        )
    return zip_path


class _ChunkAttemptError(RuntimeError):
    def __init__(self, message, log_tail=""):
        super().__init__(message)
        self.log_tail = log_tail


def _run_single_chunk(chunk_df, kaggle_username, kaggle_key, chunk_index,
                       timeout_minutes=25, max_attempts=3):
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = kaggle_username
    env["KAGGLE_KEY"] = kaggle_key

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _attempt_chunk_once(chunk_df, kaggle_username, kaggle_key,
                                        chunk_index, attempt, timeout_minutes, env)
        except _ChunkAttemptError as e:
            last_error = e
            if attempt < max_attempts and _is_manifest_not_found_signature(e.log_tail):
                print(f"Chunk {chunk_index}, attempt {attempt}: hit the known dataset-indexing "
                      f"delay signature — retrying with a longer buffer (attempt {attempt + 1}/{max_attempts}).")
                continue
            raise RuntimeError(str(e))

    raise RuntimeError(str(last_error))


def run_image_generation_chunked(manifest_path, kaggle_username, kaggle_key,
                                  chunk_size=25, progress_callback=None):
    """
    Runs image generation in chunks, saving real progress to disk after
    each one. Automatically resumes from the first incomplete chunk if
    called again on the same manifest.
    """
    df = load_and_fix_manifest(manifest_path)

    run_id = compute_run_id(manifest_path)
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    total_images = len(df)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    total_chunks = len(chunks)

    state = _load_state(run_id, manifest_path=manifest_path)
    completed = set(state.get("completed_chunks", []))

    if progress_callback and completed:
        progress_callback(len(completed), total_chunks,
                          min(len(completed) * chunk_size, total_images), total_images,
                          f"Resuming — {len(completed)}/{total_chunks} chunks already done.")

    for i, chunk_df in enumerate(chunks):
        if i in completed:
            continue

        chunk_zip = _run_single_chunk(chunk_df, kaggle_username, kaggle_key, i)

        saved_chunk_path = os.path.join(run_dir, f"chunk_{i}.zip")
        shutil.copy(chunk_zip, saved_chunk_path)

        completed.add(i)
        state["completed_chunks"] = sorted(completed)
        state["manifest_path"] = manifest_path
        _save_state(run_id, state)

        done_images = min(len(completed) * chunk_size, total_images)
        if progress_callback:
            progress_callback(len(completed), total_chunks, done_images, total_images,
                              f"Chunk {i+1}/{total_chunks} done.")

    merged_dir = os.path.join(run_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    for i in range(total_chunks):
        chunk_zip_path = os.path.join(run_dir, f"chunk_{i}.zip")
        shutil.unpack_archive(chunk_zip_path, merged_dir, "zip")

    final_zip = os.path.join(run_dir, "scene_images_batch_final")
    shutil.make_archive(final_zip, "zip", merged_dir)
    return final_zip + ".zip"


def run_image_generation_on_kaggle(manifest_path, kaggle_username, kaggle_key, timeout_minutes=60):
    """Kept for backward compatibility — single-shot version without chunking."""
    zip_path = run_image_generation_chunked(manifest_path, kaggle_username, kaggle_key)
    return zip_path, "Images generated successfully via Kaggle (chunked run)."

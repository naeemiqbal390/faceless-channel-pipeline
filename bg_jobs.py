"""
bg_jobs.py - long jobs (image generation, video build) that keep running on the SERVER.

Why: Streamlit stops a running script the moment the browser tab is closed, hidden/frozen by Chrome, or when
you click anything in the page. Everything that ran inside that script (including a 340-picture generation)
was cut off. A job started here runs in its own thread, which Streamlit never stops, and writes its progress
into a shared record that any page (even a brand-new session after a reconnect) can read.

Rules for code run inside a job: never touch st.session_state or any st.* call (there is no page attached).
The job records live in st.cache_resource, so they are shared across every browser session of this app.
They disappear only if the whole app container restarts (then Backblaze resume takes over, as before).
"""
import time
import threading
import collections
import traceback

import streamlit as st


@st.cache_resource
def _store():
    return {"lock": threading.Lock(), "jobs": {}}


def get_job(kind):
    return _store()["jobs"].get(kind)


def is_running(kind=None):
    jobs = _store()["jobs"]
    if kind:
        j = jobs.get(kind)
        return bool(j and j["status"] == "running")
    return any(j["status"] == "running" for j in jobs.values())


def start_job(kind, target, meta=None):
    """target(report, should_stop) -> result (any picklable-free python object).
    report(done, total, message) records progress. should_stop() is True after Stop was pressed.
    Returns (job, started). started is False if a job of this kind is already running."""
    store = _store()
    with store["lock"]:
        cur = store["jobs"].get(kind)
        if cur and cur["status"] == "running":
            return cur, False
        job = {
            "kind": kind, "status": "running", "done": 0, "total": 0, "msg": "Starting...",
            "log": collections.deque(maxlen=250), "started": time.time(), "finished": None,
            "updated": time.time(), "meta": dict(meta or {}), "result": None, "error": "",
            "stop": False, "id": f"{kind}-{int(time.time() * 1000)}",
        }
        store["jobs"][kind] = job

    def report(done, total, message):
        job["done"], job["total"], job["msg"] = done, total, message
        job["log"].append(time.strftime("%H:%M:%S") + "  " + str(message))
        job["updated"] = time.time()

    def run():
        try:
            job["result"] = target(report, lambda: job["stop"])
            job["status"] = "stopped" if job["stop"] else "done"
        except Exception as e:                      # never let a job die silently
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"[:600]
            job["log"].append("ERROR " + job["error"])
            print(traceback.format_exc(), flush=True)
        finally:
            job["finished"] = time.time()
            job["updated"] = time.time()

    threading.Thread(target=run, daemon=True, name=f"job-{kind}").start()
    return job, True


def request_stop(kind):
    j = get_job(kind)
    if j and j["status"] == "running":
        j["stop"] = True
        j["log"].append(time.strftime("%H:%M:%S") + "  Stop requested - finishing the picture in progress...")


def clear_job(kind):
    with _store()["lock"]:
        j = _store()["jobs"].get(kind)
        if j and j["status"] != "running":
            _store()["jobs"].pop(kind, None)


# ------------------------------------------------------------------ ready-made job bodies
def images_job_target(manifest_path, accounts, style, project_id):
    """Generate every missing picture, upload each to Backblaze in the background, then make sure
    everything local is also saved there. Returns {'images_dir', 'zip_path', 'failed'}."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import cloudflare_runner as cr
    import b2_storage

    def target(report, should_stop):
        use_b2 = bool(project_id) and b2_storage.is_configured()
        pool = ThreadPoolExecutor(max_workers=3) if use_b2 else None

        def upload(path):
            for _ in range(3):
                try:
                    if b2_storage.upload_file(
                            path, f"projects/{project_id}/scene_images/{os.path.basename(path)}"):
                        return
                except Exception:
                    pass
                time.sleep(1)

        def on_saved(scene_id, path):
            if pool:
                pool.submit(upload, path)

        images_dir, zip_path, failed = cr.run_image_generation(
            manifest_path, progress_callback=report, style=style, on_image_saved=on_saved,
            accounts=accounts, should_stop=should_stop)
        if pool:
            pool.shutdown(wait=True)
            try:
                prefix = f"projects/{project_id}/scene_images/"
                have = {os.path.basename(k) for k in b2_storage.list_keys(prefix)}
                for f in sorted(os.listdir(images_dir)):
                    if f.lower().endswith(".png") and f not in have:
                        for _ in range(3):
                            if b2_storage.upload_file(os.path.join(images_dir, f), prefix + f):
                                break
            except Exception:
                pass
        return {"images_dir": images_dir, "zip_path": zip_path, "failed": list(failed)}

    return target


def video_job_target(manifest_path, images_dir, audio_path, output_path):
    def target(report, should_stop):
        import video_assembler

        def cb(done, total, message):
            report(done, total, f"{done}/{total} scenes - {message}")

        final_path, filled = video_assembler.assemble_video(
            manifest_path, images_dir, audio_path, output_path, progress_callback=cb)
        return {"path": final_path, "filled": list(filled or [])}

    return target

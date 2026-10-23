"""
scene_fixer.py — review and repair single scene images BEFORE the video is built.

- parse_scene_numbers("12, 57, 100-103")  -> [12, 57, 100, 101, 102, 103]
- regenerate_scenes(...)  replaces scene_NNN.png in place (same slot, same file name), so the
  video uses it at exactly the same position.
    * a scene's FIRST fix = same prompt, fresh random seed (most bad pictures are just bad luck);
    * its SECOND, THIRD... fix = a reworded prompt (new framing, hands/close-ups simplified).
  Only the scenes you ask for are touched; scenes that fail keep their old picture.
- Per-scene history is kept in fixes.json next to the images, so "this scene was already fixed once"
  survives page reruns.
"""
import os
import re
import json
import time

import cloudflare_runner as cr

THUMB_W = 360


# ---------------------------------------------------------------- parsing
def parse_scene_numbers(text, valid_ids=None):
    """Returns (sorted_unique_ids, problems). Accepts commas, spaces, ranges like 5-9."""
    ids, problems = set(), []
    for tok in re.split(r"[,\s;]+", (text or "").strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            if b - a > 400:
                problems.append(f"Range too large: {tok}")
                continue
            ids.update(range(a, b + 1))
        elif tok.isdigit():
            ids.add(int(tok))
        else:
            problems.append(f"Not a scene number: {tok}")
    out = sorted(ids)
    if valid_ids is not None:
        valid = set(valid_ids)
        bad = [i for i in out if i not in valid]
        if bad:
            problems.append("No such scene: " + ", ".join(map(str, bad[:10])))
        out = [i for i in out if i in valid]
    return out, problems


# ---------------------------------------------------------------- fixes.json
def _run_dir(images_dir):
    return os.path.dirname(images_dir)


def _fixes_path(images_dir):
    return os.path.join(_run_dir(images_dir), "fixes.json")


def load_fixes(images_dir):
    try:
        with open(_fixes_path(images_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_fixes(images_dir, fixes):
    try:
        with open(_fixes_path(images_dir), "w", encoding="utf-8") as f:
            json.dump(fixes, f)
    except Exception:
        pass


# ---------------------------------------------------------------- rewording
_HAND_WORDS = re.compile(
    r"\b(hands?|fingers?|arms?|holding|holds|grips?|gripping|hugs?|hugging|wraps|wrapping|"
    r"touching|pressed|pressing|clasped|raised|raises|waving|wave|pointing|points)\b", re.I)


def local_reword(prompt, attempt):
    """Offline fallback: drop hand/contact clauses, widen the shot, simplify. attempt >= 1."""
    text = re.sub(r"\bclose-?up of\b", "Medium shot of", prompt, flags=re.I)
    text = re.sub(r"\bextreme close-?up\b", "medium shot", text, flags=re.I)
    clauses = [c.strip() for c in re.split(r",\s*", text)]
    kept = [c for c in clauses if not _HAND_WORDS.search(c)]
    text = ", ".join(kept) if len(kept) >= 2 else text
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    if attempt <= 1:
        lead = "Medium wide shot, one clear face per person, hands resting naturally at the sides. "
    elif attempt == 2:
        lead = "Wide shot from a distance, small simple figures standing apart, hands out of view. "
    else:
        lead = "A single figure seen from behind in a simple wide scene, no one touching. "
    return lead + text + "."


_REWORD_SYSTEM = (
    "You repair text-to-image prompts whose pictures came out malformed (extra arms or limbs, a doubled "
    "face). Rewrite the scene so the SAME story moment, setting and characters stay, but the composition "
    "is safer: medium or wide shot, no extreme close-ups of faces, at most two people who stand apart "
    "(no hugging, overlapping or touching), hands simple and resting, in pockets or out of frame, and no "
    "fiddly held objects. Keep every character description word-for-word as written. No style words, no "
    "text or numbers in the picture. 20 to 45 words. "
    'Reply with JSON only: {"prompt": "..."}'
)


def reword_prompt(prompt, attempt, gemini_key=None, gemini_model="", gemini_fn=None):
    """Returns (new_prompt, how). Uses Gemini when available, otherwise the offline rules."""
    if gemini_key and gemini_fn:
        extra = ""
        if attempt >= 2:
            extra = " This is a repeat failure: simplify more strongly (one or two small figures, wide shot)."
        try:
            data, _err = gemini_fn(_REWORD_SYSTEM, f"Original prompt: {prompt}{extra}", gemini_key, gemini_model)
            cand = (data or {}).get("prompt") if isinstance(data, dict) else None
            if cand and 40 <= len(cand) <= 900:
                return cand.strip(), "Gemini reword"
        except Exception:
            pass
    return local_reword(prompt, attempt), "offline reword"


# ---------------------------------------------------------------- thumbnails for review
def make_thumb(image_path, cache_dir, width=THUMB_W):
    """Small JPEG copy (rebuilt only when the original changes). Returns its path or the original."""
    try:
        from PIL import Image
        os.makedirs(cache_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        t = os.path.join(cache_dir, f"{stem}.jpg")
        if os.path.exists(t) and os.path.getmtime(t) >= os.path.getmtime(image_path):
            return t
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            h = max(1, int(im.height * width / im.width))
            im.resize((width, h)).save(t, "JPEG", quality=82)
        return t
    except Exception:
        return image_path


# ---------------------------------------------------------------- regeneration
def regenerate_scenes(manifest_path, scene_ids, style, accounts, mode="auto",
                      gemini_key=None, gemini_model="", gemini_fn=None,
                      progress=None, on_saved=None):
    """
    Regenerates the listed scenes, replacing scene_NNN.png in place.
    mode: "auto" = first fix same prompt / later fixes reworded;  "reword" = always reworded.
    Returns {scene_id: {"ok": bool, "how": str, "prompt": str, "error": str}}.
    """
    say = progress or (lambda msg: None)
    df = cr._load_manifest_df(manifest_path)
    prompts = {}
    for _, row in df.iterrows():
        try:
            prompts[int(row["scene_id"])] = str(row["prompt"]) if row["prompt"] == row["prompt"] else "drawing"
        except Exception:
            pass
    images_dir = cr.get_images_dir(manifest_path)
    fixes = load_fixes(images_dir)

    cfg = cr.STYLE_PRESETS.get(style, cr.STYLE_PRESETS[cr.DEFAULT_STYLE])
    negative = cfg.get("negative", "") + ", " + cr.UNIVERSAL_NEGATIVE

    pool = cr._AccountPool(accounts)
    model_state = cr.ModelState(list(cr.MODEL_CANDIDATES))
    limiter = cr.AdaptiveRateLimiter()
    results = {}

    for sid in scene_ids:
        if sid not in prompts:
            results[sid] = {"ok": False, "how": "", "prompt": "", "error": "scene not in manifest"}
            continue
        rec = fixes.get(str(sid), {"attempts": 0})
        attempts = int(rec.get("attempts", 0))
        original = prompts[sid]
        if mode == "reword" or attempts >= 1:
            level = max(1, attempts if mode != "reword" else attempts + 1)
            base, how = reword_prompt(original, level, gemini_key, gemini_model, gemini_fn)
        else:
            base, how = original, "same prompt, new seed"
        say(f"Scene {sid}: trying again ({how}).")

        out_path = os.path.join(images_dir, f"scene_{sid:03d}.png")
        tmp_path = os.path.join(images_dir, f"scene_{sid:03d}.new.png")
        soften = 0
        done = False
        err = ""
        while not done:
            acct = pool.current()
            if acct is None:
                err = "all Cloudflare accounts are out of daily quota (resets 05:00 Pakistan time)"
                break
            label, acct_id, acct_token = acct
            use = base if not soften else cr._soften_prompt(base, soften)
            try:
                limiter.wait()
                cr._generate_with_fallback(acct_id, acct_token, model_state, cr.compose_prompt(cfg, use), negative,
                                           tmp_path, limiter)
                os.replace(tmp_path, out_path)      # swap in only after a good image exists
                done = True
            except cr.AuthenticationError as e:
                pool.retire("credentials rejected", str(e))
                err = "credentials rejected"
            except cr.ContentFilterError as e:
                soften += 1
                err = "blocked by content filter"
                if soften > 3:
                    break
            except Exception as e:
                if "daily neuron quota exhausted" in str(e).lower():
                    pool.retire("daily quota used up", str(e))
                    err = "daily quota used up"
                    continue
                err = str(e)[:200]
                break
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        if done:
            fixes[str(sid)] = {"attempts": attempts + 1, "prompt": base, "how": how}
            _save_fixes(images_dir, fixes)
            results[sid] = {"ok": True, "how": how, "prompt": base, "error": ""}
            if on_saved:
                try:
                    on_saved(sid, out_path)
                except Exception:
                    pass
            say(f"Scene {sid}: new picture saved.")
        else:
            results[sid] = {"ok": False, "how": how, "prompt": base, "error": err}
            say(f"Scene {sid}: could not be redone ({err}); the old picture is kept.")
            if pool.exhausted:
                for rest in scene_ids[scene_ids.index(sid) + 1:]:
                    results[rest] = {"ok": False, "how": "", "prompt": "", "error": err}
                break
    return results

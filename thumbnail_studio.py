"""
thumbnail_studio.py
Helpers for the Thumbnail tab. Kept separate from app.py so the working video
pipeline is not touched. Everything here is plain Python (no Streamlit).

  rank_scene_images()      picks the strongest-looking scene images for a thumbnail
  suggest_text_ideas()     2-3 short on-image text ideas (Gemini, rule-based fallback)
  suggest_image_prompts()  3 distinct concepts for NEW thumbnail pictures
  generate_new_pictures()  renders them on Cloudflare, accounts used in sequence
  enhance_picture()        gentle contrast/colour boost so thumbnails pop
  quality_report()         readability warnings (text length, dark/flat picture)
"""

import json
import os
import re
import time

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
MAX_TEXT_WORDS = 4

# --------------------------------------------------------------------------
# 1. Rank existing scene images
# --------------------------------------------------------------------------

def _score_image(path):
    """Heuristic 0-100: good brightness, contrast, colour, and a clear centre subject."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return 0.0, {}
    im.thumbnail((256, 256))
    gray = im.convert("L")
    mean = ImageStat.Stat(gray).mean[0]
    std = ImageStat.Stat(gray).stddev[0]

    # colourfulness (Hasler-Susstrunk), 0..~100+
    r, g, b = [ImageStat.Stat(c) for c in im.split()]
    # approximate with channel means/stddevs
    rg_std = abs(r.mean[0] - g.mean[0])
    yb_std = abs(0.5 * (r.mean[0] + g.mean[0]) - b.mean[0])
    colourful = (rg_std ** 2 + yb_std ** 2) ** 0.5 + 0.3 * ((r.stddev[0] + g.stddev[0] + b.stddev[0]) / 3)

    # subject focus: edge energy in the centre vs the border
    edges = gray.filter(ImageFilter.FIND_EDGES)
    w, h = edges.size
    centre = edges.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    c_energy = ImageStat.Stat(centre).mean[0]
    all_energy = ImageStat.Stat(edges).mean[0] + 1e-6
    focus = min(c_energy / all_energy, 2.0)       # >1 means detail concentrated in the middle

    bright_score = max(0.0, 1.0 - abs(mean - 125) / 125)          # best around mid-bright
    contrast_score = min(std / 70.0, 1.0)
    colour_score = min(colourful / 60.0, 1.0)
    focus_score = min(focus / 1.4, 1.0)
    detail_score = min(all_energy / 12.0, 1.0)                    # not empty / not noise-flat

    total = 100 * (0.22 * bright_score + 0.28 * contrast_score + 0.18 * colour_score
                   + 0.20 * focus_score + 0.12 * detail_score)
    return total, {"brightness": round(mean), "contrast": round(std), "colour": round(colourful)}


def rank_scene_images(scene_files, top_n=6, min_gap=None):
    """
    scene_files: {scene_number: path}. Returns [(scene_number, path, score)] best first,
    skipping neighbours so the options differ (min_gap scenes apart).
    """
    if not scene_files:
        return []
    nums = sorted(scene_files)
    if min_gap is None:
        min_gap = max(3, len(nums) // 25)
    scored = []
    for n in nums:
        s, _ = _score_image(scene_files[n])
        scored.append((n, scene_files[n], s))
    scored.sort(key=lambda x: -x[2])
    picked = []
    for n, p, s in scored:
        if all(abs(n - q[0]) >= min_gap for q in picked):
            picked.append((n, p, round(s, 1)))
        if len(picked) >= top_n:
            break
    return picked

# --------------------------------------------------------------------------
# 2. Gemini helper
# --------------------------------------------------------------------------

def _gemini_json(system_prompt, user_prompt, api_key, model_override=""):
    """Returns (parsed_json_or_None, error_message_or_None)."""
    if not api_key:
        return None, "No GEMINI_API_KEY in Secrets."
    models = ([model_override] if model_override else []) + [m for m in GEMINI_MODELS if m != model_override]
    last = "unknown error"
    for model in models:
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.9, "responseMimeType": "application/json"},
        }
        for attempt in range(2):
            try:
                r = requests.post(GEMINI_URL.format(model=model), json=body, timeout=60,
                                  headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
            except Exception as e:
                last = f"{model}: network error {type(e).__name__}"
                time.sleep(2)
                continue
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts).strip()
                    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
                    return json.loads(text), None
                except Exception:
                    last = f"{model}: unreadable response"
                    break
            last = f"{model}: HTTP {r.status_code}"
            if r.status_code in (401, 403):
                return None, f"Gemini rejected the API key ({r.status_code})."
            if r.status_code == 429:
                time.sleep(3)
                continue
            break
    return None, last

# --------------------------------------------------------------------------
# 3. Text ideas
# --------------------------------------------------------------------------

_TEXT_SYSTEM = f"""You write the SHORT on-image text for YouTube thumbnails on a faceless, narrated finance/story channel (English-speaking adults, 20+).
Rules:
- 2 to {MAX_TEXT_WORDS} words, ALL information readable at phone size. No full sentences.
- The text must ADD to the video title, not repeat it (title says the topic; thumbnail adds tension, a number, or a consequence).
- Curiosity or consequence, but truthful to the video. No guarantees, no get-rich claims.
- Give each idea a different angle (e.g. consequence, number, question, warning).
- "keyword" is the single most important word from the text (it will be shown in red).
Return ONLY JSON: {{"ideas":[{{"text":str,"keyword":str,"why":str}}, ...3 items]}}"""


def _clean_idea(it):
    if not isinstance(it, dict):
        return None
    text = re.sub(r"\s+", " ", str(it.get("text", "")).strip().strip('"')).upper()
    words = text.split()
    if not (1 <= len(words) <= MAX_TEXT_WORDS + 1):
        return None
    kw = str(it.get("keyword", "")).strip().upper()
    if kw and kw not in text:
        kw = ""
    if not kw:
        kw = max(words, key=len)
    return {"text": " ".join(words[:MAX_TEXT_WORDS + 1]), "keyword": kw, "why": str(it.get("why", ""))[:160]}


def fallback_text_ideas(title):
    """No-AI backup so the box is never empty: built from the title itself."""
    words = [w.strip(".,!?:;\"'()") for w in re.split(r"\s+", title or "") if w.strip()]
    nums = [w for w in words if re.search(r"\d", w)]
    ideas = []
    if nums:
        ideas.append({"text": f"{nums[0]}... THEN WHAT?".upper(), "keyword": nums[0].upper(), "why": "Number plus open question."})
    ideas.append({"text": "NOBODY WARNS YOU", "keyword": "NOBODY", "why": "Hidden-truth angle."})
    ideas.append({"text": "THE REAL COST", "keyword": "COST", "why": "Consequence angle."})
    ideas.append({"text": "IT GETS WORSE", "keyword": "WORSE", "why": "Tension angle."})
    return ideas[:3]


def suggest_text_ideas(title, synopsis, api_key, model_override=""):
    """Returns (ideas, note). ideas always has 1-3 items."""
    prompt = f"Video title: {title}\nVideo opening/summary: {synopsis[:1200]}\nWrite 3 thumbnail text ideas."
    data, err = _gemini_json(_TEXT_SYSTEM, prompt, api_key, model_override)
    ideas = []
    if data:
        raw = data.get("ideas", []) if isinstance(data, dict) else data
        ideas = [i for i in (_clean_idea(x) for x in (raw or [])) if i][:3]
    if ideas:
        return ideas, None
    return fallback_text_ideas(title), (f"Used simple built-in ideas ({err})." if err else "Used simple built-in ideas.")

# --------------------------------------------------------------------------
# 4. New picture concepts + generation
# --------------------------------------------------------------------------

_PROMPT_SYSTEM = """You design picture concepts for YouTube thumbnails. Each concept is rendered by an image model in a fixed illustration style, so describe ONLY the scene.
Rules for each of the 3 concepts:
- ONE clear subject close to the camera, centred, large in frame (it will be cropped to wide 16:9 from the centre).
- An emotion or consequence visible at a glance (worry, shock, relief, a heavy choice). Strong simple colour contrast.
- Simple background; leave calm space on one side for text. No text, letters, numbers, logos or signs in the picture.
- 25 to 45 words. Self-contained: describe the character fully each time, do not refer to other concepts.
- Three clearly different ideas (e.g. emotional close-up, before/after contrast, a symbolic object).
Return ONLY JSON: {"concepts":[{"label":str,"prompt":str}, ...3 items]}"""


def fallback_image_prompts(title):
    t = (title or "a money decision").strip()
    return [
        {"label": "Worried close-up", "prompt": f"Close-up of a worried young adult sitting at a kitchen table, holding a single bill, warm lamp light, strong contrast, simple background, calm empty space on the right. Mood of the story: {t[:80]}"},
        {"label": "Heavy choice", "prompt": f"A young adult standing at a fork in a road, one path bright and one dim, centred and large in frame, simple background, calm empty space on the left. Mood of the story: {t[:80]}"},
        {"label": "Symbol", "prompt": f"One oversized old piggy bank cracked open on a table with a few coins spilling, dramatic side light, strong colour contrast, simple background, empty space on the right. Mood of the story: {t[:80]}"},
    ]


def suggest_image_prompts(title, synopsis, api_key, model_override=""):
    prompt = f"Video title: {title}\nVideo opening/summary: {synopsis[:1200]}\nWrite 3 thumbnail picture concepts."
    data, err = _gemini_json(_PROMPT_SYSTEM, prompt, api_key, model_override)
    out = []
    if data:
        raw = data.get("concepts", []) if isinstance(data, dict) else data
        for x in raw or []:
            if isinstance(x, dict) and str(x.get("prompt", "")).strip():
                out.append({"label": str(x.get("label", "Concept"))[:40], "prompt": str(x["prompt"]).strip()[:600]})
    if len(out) >= 3:
        return out[:3], None
    return fallback_image_prompts(title), (f"Used simple built-in concepts ({err})." if err else "Used simple built-in concepts.")


def generate_new_pictures(concepts, style_prefix, accounts, out_dir, attempt_fn, classify_fn, soften_fn, negative=""):
    """
    Renders each concept on Cloudflare. Accounts are used strictly in order; an account
    whose daily quota is gone is skipped. A content-filter block is retried with a
    reworded (softened) prompt, up to 3 levels.
    Returns (results, notes) where results = [(label, path)] and notes = [strings].
    """
    os.makedirs(out_dir, exist_ok=True)
    model = "@cf/black-forest-labs/flux-1-schnell"
    results, notes = [], []
    idx = 0                                         # current account index
    stamp = int(time.time())
    for n, c in enumerate(concepts, 1):
        done = False
        level = 0
        while not done:
            if idx >= len(accounts):
                notes.append("All Cloudflare accounts are out of quota or rejected; stopped early.")
                return results, notes
            label, acct_id, token = accounts[idx]
            prompt = style_prefix + (soften_fn(c["prompt"], level) if level else c["prompt"])
            out = os.path.join(out_dir, f"thumb_new_{stamp}_{n}.png")
            try:
                attempt_fn(acct_id, token, model, prompt, negative, out)
                results.append((c["label"], out))
                done = True
            except Exception as e:
                kind = classify_fn(str(e))
                if kind == "quota_exhausted":
                    notes.append(f"{label}: daily quota used up, moving to the next account.")
                    idx += 1
                elif kind == "auth_error":
                    notes.append(f"{label}: credentials rejected, moving to the next account.")
                    idx += 1
                elif kind == "content_filtered" and level < 3:
                    level += 1
                    notes.append(f"'{c['label']}': prompt reworded (level {level}) after a content-filter block.")
                else:
                    notes.append(f"'{c['label']}' failed: {str(e)[:140]}")
                    break
    return results, notes

# --------------------------------------------------------------------------
# 5. Polish + checks
# --------------------------------------------------------------------------

def enhance_picture(src_path, out_path):
    """Gentle punch: a little more contrast, colour and sharpness. Returns out_path."""
    im = Image.open(src_path).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.12)
    im = ImageEnhance.Color(im).enhance(1.18)
    im = ImageEnhance.Sharpness(im).enhance(1.15)
    im.save(out_path, "PNG")
    return out_path


def quality_report(thumb_path, text):
    """Returns a list of plain-language warnings (empty list = looks fine)."""
    warns = []
    words = [w for w in (text or "").split() if w]
    if len(words) > MAX_TEXT_WORDS:
        warns.append(f"Text has {len(words)} words. {MAX_TEXT_WORDS} or fewer reads better on a phone.")
    if len(text or "") > 26:
        warns.append("Text is long; letters will be small. Try shortening it.")
    try:
        im = Image.open(thumb_path).convert("L")
        im.thumbnail((200, 120))
        st = ImageStat.Stat(im)
        if st.mean[0] < 45:
            warns.append("Picture is very dark; it may disappear on YouTube's dark theme.")
        if st.stddev[0] < 25:
            warns.append("Picture has low contrast; it may look flat in small sizes.")
    except Exception:
        pass
    return warns

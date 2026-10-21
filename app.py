"""
Title Studio — a small, separate Streamlit app that serves ONE video title at a time.

  Use this  -> saved as "used", never offered again, shown in the Used list
  Skip      -> saved as "skipped", never offered again, next title appears

Titles come from Gemini (free text tier). One Gemini call makes ~10 candidates;
the app filters them (length, banned claims, duplicates of anything you already
used/skipped) and serves the best-scored one first. History is saved to
Backblaze B2 (same bucket as your pipeline, under titles/) if B2 secrets exist,
otherwise to a local file that is lost on reboot.

Secrets (Streamlit Cloud > Settings > Secrets):
  GEMINI_API_KEY = "..."            (required)
  GEMINI_MODEL   = "gemini-2.5-flash"   (optional override)
  B2_KEY_ID / B2_APPLICATION_KEY / B2_BUCKET_NAME / B2_ENDPOINT_URL  (optional)
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests
import streamlit as st

st.set_page_config(page_title="Title Studio", page_icon="🎯", layout="centered")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
GEMINI_FALLBACK = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]   # last resort only; names get retired
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BATCH_SIZE = 8
MAX_TITLE_CHARS = 70
SIMILARITY_LIMIT = 0.72          # reject titles this similar to anything already seen
HISTORY_KEY = "titles/history.json"
LOCAL_HISTORY = "/tmp/title_history.json"

FORMATS = {
    "Any (best mix)": "Any of the formats below; choose what fits best.",
    "Your Life at Every Level": "Tier progression told in second person: 'Your Life at Every Level of ___'.",
    "POV story": "Second-person story: 'POV: You ___ — Everything Changed'. A fictional 'you' with real consequences.",
    "Money mistake story": "A named or unnamed fictional person makes one money decision; the title hints at the cost.",
    "Milestone / what changes": "'How Everything Changes After ___' (a savings or net-worth milestone).",
    "Explained / history": "Crashes, bubbles, scams, famous money stories explained simply.",
    "Two paths": "Two fictional people, same starting point, different choices.",
}

PILLARS = [
    "saving money and savings milestones",
    "debt, credit scores and interest",
    "investing basics and compound growth",
    "money psychology and spending habits",
    "alternative funding (grants, crowdfunding, microloans, community lenders)",
    "financial history: crashes, bubbles, scams",
    "side hustles and small-business money decisions",
    "everyday money traps (subscriptions, car payments, lifestyle creep)",
]

# Pattern library: structures that worked for reference channels in this niche.
# They are given to the model as PATTERNS to learn from, never to copy.
PATTERN_EXAMPLES = [
    "Your Life at Every Level of Credit Score (300 to 850)",
    "Your Life at Every Level of Compound Interest",
    "A Day in the Life of a Quiet Millionaire",
    "POV: You Chose Wealth Over Looking Rich",
    "How Everything Changes After You Save $10,000",
    "The Car Payment Trap Nobody Talks About",
    "Who Can Actually Afford a $500K House (The Math Is Brutal)",
    "Do This EVERY Time You Get Paid (Payday Routine)",
    "The 2008 Financial Crisis Explained Like You're 5",
    "The $63 BILLION Company That Sold Nothing",
    "The Worst Time to be Rich as a Human",
    "How To Survive a Layoff With Six Weeks of Savings",
    "Surviving $50,000 of Debt: The First Year",
    "Debt Survival: Month by Month",
    "Your First Year After You Quit Your Job",
    "Is the 50/30/20 Rule Real?",
    "What Happens to Your Money If the Bank Fails?",
]

BANNED = re.compile(
    r"guarantee|risk[- ]free|get rich quick|double your money|100% (safe|profit|return)|"
    r"can't lose|cannot lose|secret loophole|financial advice|buy now|"
    r"you will (be|become) (a )?millionaire",
    re.I,
)

SYSTEM_PROMPT = f"""You write YouTube video titles for a faceless, narrated, illustrated finance channel.
Audience: English-speaking adults, 20+ (US/UK style English).
Videos are 8-15 minute stories and explainers: personal finance, saving, building wealth,
alternative funding, and money history, often told as a fictional person's story with real,
realistic consequences and accurate numbers.

Your job: produce titles with a high chance of being clicked AND watched, in the style of this niche.

What makes a title work here:
- One clear promise or reveal (a transformation, a consequence, a hidden cost, a comparison).
- Curiosity gap: the viewer must feel they need the answer, but the title must be TRUE to the video.
- Specific beats vague: a concrete number, stage, or situation ($10,000, 30 years, "every level").
- Second person or a single clear character works well. Plain words, no jargon.
- Short plain titles that name a situation and a time span or amount ("X Survival 1 Year", "How To Survive Y") also work well; do not force every title into a curiosity-gap shape.
- Under {MAX_TITLE_CHARS} characters. At most one word in ALL CAPS. No emoji, no hashtags.
- The story must be deliverable by a narrated video with illustrated scenes (no charts needed to make the point).

Hard rules:
- Never promise guaranteed results, risk-free returns, get-rich-quick outcomes, or give personal financial advice.
- Numbers must be plausible and illustrative; never invent fake statistics presented as fact.
- Do not copy any example title; learn their structure and create new, original ones.
- Every title in the batch must use a clearly different idea, not rewordings of each other.

Pattern examples (structures only, do NOT reuse): {json.dumps(PATTERN_EXAMPLES)}

Return ONLY a JSON array. Each item: {{"title": str, "format": str, "hook": str, "why": str, "score": int}}
 - hook: the curiosity trigger in 6 words or fewer
 - why: one short sentence on why a viewer would click
 - score: 1-100, your honest estimate of click-and-watch strength (be critical; use the whole range)
"""

# --------------------------------------------------------------------------
# Storage (B2 optional, local fallback)
# --------------------------------------------------------------------------

def _secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


def _b2_ready():
    return all(_secret(k) for k in ["B2_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME", "B2_ENDPOINT_URL"])


@st.cache_resource
def _b2_client_for(endpoint, key_id, app_key):
    """Created once and reused (importing boto3 and building a client on every click was slow)."""
    import boto3
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key_id, aws_secret_access_key=app_key)


def _b2_params():
    return (_secret("B2_ENDPOINT_URL"), _secret("B2_KEY_ID"), _secret("B2_APPLICATION_KEY"), _secret("B2_BUCKET_NAME"))


def _b2_client():
    endpoint, key_id, app_key, _ = _b2_params()
    return _b2_client_for(endpoint, key_id, app_key)


def empty_history():
    return {"used": [], "skipped": []}


def load_history():
    """Returns (history_dict, storage_label)."""
    if _b2_ready():
        try:
            obj = _b2_client().get_object(Bucket=_secret("B2_BUCKET_NAME"), Key=HISTORY_KEY)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            data.setdefault("used", [])
            data.setdefault("skipped", [])
            return data, "Backblaze B2 (permanent)"
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                return empty_history(), "Backblaze B2 (permanent)"
            # fall through to local if B2 had a real error
    if os.path.exists(LOCAL_HISTORY):
        try:
            with open(LOCAL_HISTORY, encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("used", [])
            data.setdefault("skipped", [])
            return data, "local file (lost on reboot)"
        except Exception:
            pass
    return empty_history(), ("Backblaze B2 (permanent)" if _b2_ready() else "local file (lost on reboot)")


def _save_to_b2(payload, client, bucket, lock, status):
    """Runs in a background thread so Skip / Use This never wait on the network.
    Everything it needs is passed in (the thread has no Streamlit context)."""
    with lock:
        ok = False
        for _ in range(3):
            try:
                client.put_object(Bucket=bucket, Key=HISTORY_KEY, Body=payload.encode("utf-8"),
                                  ContentType="application/json")
                ok = True
                break
            except Exception:
                time.sleep(1)
        status["b2_save_ok"] = ok


def save_history(history):
    """Local file is written instantly; the B2 copy is uploaded in the background."""
    payload = json.dumps(history, ensure_ascii=False, indent=1)
    try:
        with open(LOCAL_HISTORY, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass
    if _b2_ready():
        try:
            client = _b2_client()
            bucket = _secret("B2_BUCKET_NAME")
            lock = _MODEL_CACHE.setdefault("save_lock", threading.Lock())
            threading.Thread(target=_save_to_b2, args=(payload, client, bucket, lock, _MODEL_CACHE),
                             daemon=True).start()
        except Exception:
            _MODEL_CACHE["b2_save_ok"] = False
    return True


# --------------------------------------------------------------------------
# Title logic
# --------------------------------------------------------------------------

def normalize(t):
    return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()


def too_similar(title, seen_titles):
    n = normalize(title)
    for s in seen_titles:
        if SequenceMatcher(None, n, normalize(s)).ratio() >= SIMILARITY_LIMIT:
            return True
    return False


def clean_candidate(item):
    """Validate one model item. Returns cleaned dict or None."""
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "")).strip().strip('"').strip()
    if not title or len(title) > MAX_TITLE_CHARS or len(title) < 12:
        return None
    if BANNED.search(title):
        return None
    caps_words = [w for w in re.findall(r"[A-Za-z']+", title) if len(w) > 2 and w.isupper()]
    if len(caps_words) > 1:
        return None
    try:
        score = int(item.get("score", 50))
    except Exception:
        score = 50
    return {
        "title": title,
        "format": str(item.get("format", ""))[:40],
        "hook": str(item.get("hook", ""))[:60],
        "why": str(item.get("why", ""))[:200],
        "score": max(1, min(100, score)),
    }


def parse_batch(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(data, dict):
        data = data.get("titles", [])
    return data if isinstance(data, list) else []


@st.cache_resource
def _get_model_cache():
    """One dict that survives Streamlit's re-run of this script on every click."""
    return {}


_MODEL_CACHE = _get_model_cache()


def discover_models(api_key):
    """Ask Google which Gemini 'flash' models can be used right now (newest stable first).
    Model names get retired, so a fixed list goes stale. Returns [] on any failure."""
    if "list" in _MODEL_CACHE:
        return _MODEL_CACHE["list"]
    found = []
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"pageSize": 200}, headers={"x-goog-api-key": api_key}, timeout=8)
        if r.status_code == 200:
            skip = ("image", "tts", "audio", "live", "embed", "robot", "computer", "vision", "learnlm", "gemma")
            rows = []
            for m in r.json().get("models", []):
                name = str(m.get("name", "")).replace("models/", "")
                if not name.startswith("gemini") or "flash" not in name:
                    continue
                if any(t in name for t in skip):
                    continue
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
                version = float(ver.group(1)) if ver else 0.0
                unstable = any(t in name for t in ("preview", "exp", "-0"))
                lite = "lite" in name
                rows.append((-version, unstable, lite, name))
            rows.sort()
            found = [x[3] for x in rows]
    except Exception:
        found = []
    if found:
        _MODEL_CACHE["list"] = found
    return found


def candidate_models(api_key, override=""):
    """Override first, then Google's self-updating alias, then whatever Google lists, then old names."""
    out = []
    for m in [override, "gemini-flash-latest"] + discover_models(api_key)[:5] + ["gemini-flash-lite-latest"] + GEMINI_FALLBACK:
        if m and m not in out:
            out.append(m)
    return out


def call_gemini(user_prompt):
    """Returns (text, model_used, error_message). Tuned for speed: remembers the model that worked,
    turns off 'thinking' (slow, and not needed for titles), fails over without waiting."""
    key = _secret("GEMINI_API_KEY")
    if not key:
        return None, None, "GEMINI_API_KEY is missing from Secrets."
    models = candidate_models(key, _secret("GEMINI_MODEL") or "")
    good = _MODEL_CACHE.get("good")
    if good in models:                       # last model that worked goes first
        models = [good] + [m for m in models if m != good]
    errors = []
    for model in models:
        use_thinking_off = _MODEL_CACHE.get("no_think_ok", True)
        for attempt in range(2):
            gen = {"temperature": 1.0, "responseMimeType": "application/json", "maxOutputTokens": 2048}
            if use_thinking_off:
                gen["thinkingConfig"] = {"thinkingBudget": 0}
            body = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": gen,
            }
            try:
                r = requests.post(
                    API_URL.format(model=model), json=body,
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    timeout=40,
                )
            except Exception as e:
                errors.append(f"{model}: network error {type(e).__name__}")
                break
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    _MODEL_CACHE["good"] = model
                    _MODEL_CACHE["no_think_ok"] = use_thinking_off
                    return "".join(p.get("text", "") for p in parts), model, None
                except Exception:
                    errors.append(f"{model}: empty or blocked response")
                    break
            if r.status_code == 400 and use_thinking_off and "think" in r.text.lower():
                use_thinking_off = False     # this model doesn't accept the setting: retry once without it
                continue
            errors.append(f"{model}: HTTP {r.status_code} {r.text[:120]}")
            if r.status_code in (401, 403):
                return None, None, f"Gemini rejected the API key ({r.status_code}). Check GEMINI_API_KEY."
            break                            # 404 / 429 / 5xx: next model right away, no waiting
    return None, None, "No Gemini model worked. " + " | ".join(e[:110] for e in errors[:3])


def generate_batch(format_name, topic, seen_titles):
    fmt_desc = FORMATS.get(format_name, FORMATS["Any (best mix)"])
    topic_line = topic.strip() if topic.strip() else "Choose varied topics from: " + "; ".join(PILLARS)
    recent = seen_titles[-60:]
    prompt = (
        f"Write {BATCH_SIZE} new titles.\n"
        f"Format wanted: {format_name} — {fmt_desc}\n"
        f"Topic/angle: {topic_line}\n"
        f"Do NOT produce anything similar to these already seen titles:\n{json.dumps(recent)}\n"
    )
    text, model, err = call_gemini(prompt)
    if err:
        return [], err, None
    items = [clean_candidate(i) for i in parse_batch(text)]
    items = [i for i in items if i]
    fresh, taken = [], list(seen_titles)
    for it in sorted(items, key=lambda x: -x["score"]):
        if not too_similar(it["title"], taken):
            fresh.append(it)
            taken.append(it["title"])
    if not fresh:
        return [], "Gemini replied, but no usable new titles passed the filters. Press the Get a title button to try again.", model
    return fresh, None, model

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def init_state():
    if "history" not in st.session_state:
        h, label = load_history()
        st.session_state.history = h
        st.session_state.storage_label = label
        st.session_state.queue = []
        st.session_state.current = None
        st.session_state.last_error = None
        st.session_state.last_model = None
        st.session_state.save_failed = False


def seen_titles():
    h = st.session_state.history
    return [x["title"] for x in h["used"]] + [x["title"] for x in h["skipped"]]


def advance(format_name, topic):
    """Make st.session_state.current the next title, fetching a batch if needed."""
    st.session_state.last_error = None
    if not st.session_state.queue:
        seen = seen_titles()
        t0 = time.time()
        with st.spinner("Writing titles (usually a few seconds)…"):
            batch, err, model = generate_batch(format_name, topic, seen)
        st.session_state.last_model = model
        st.session_state.last_seconds = round(time.time() - t0, 1)
        if err:
            st.session_state.last_error = err
            st.session_state.current = None
            return
        st.session_state.queue = batch
    st.session_state.current = st.session_state.queue.pop(0)


def record(kind, cand):
    entry = {
        "title": cand["title"], "format": cand.get("format", ""),
        "score": cand.get("score"), "hook": cand.get("hook", ""),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    st.session_state.history[kind].append(entry)
    st.session_state.save_failed = not save_history(st.session_state.history)


init_state()

st.title("🎯 Title Studio")
st.caption("One title at a time. Use it or skip it — it never comes back.")

with st.sidebar:
    st.subheader("Settings")
    format_name = st.selectbox("Format", list(FORMATS.keys()))
    topic = st.text_input("Topic or angle (optional)",
                          placeholder="e.g. credit cards, first $10,000, a bubble…")
    if st.session_state.get("last_model"):
        st.caption(f"Last batch: {st.session_state.last_model}, {st.session_state.get('last_seconds', '?')} s")
    st.caption(f"History saved to: **{st.session_state.storage_label}**")
    st.caption(f"Used: {len(st.session_state.history['used'])} · "
               f"Skipped: {len(st.session_state.history['skipped'])}")
    if not _secret("GEMINI_API_KEY"):
        st.error("Add GEMINI_API_KEY in Secrets.")

if _MODEL_CACHE.get("b2_save_ok") is False:
    st.warning("Couldn't save history to B2 just now. It's kept for this session; it will retry on the next action.")

cur = st.session_state.current

if cur is None:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    if st.button("Get a title", type="primary", use_container_width=True):
        advance(format_name, topic)
        st.rerun()
else:
    st.markdown("### Your title")
    st.code(cur["title"], language=None)          # built-in copy button
    c1, c2, c3 = st.columns(3)
    c1.metric("Strength", f"{cur['score']}/100")
    c2.metric("Format", cur["format"] or "—")
    c3.metric("Length", f"{len(cur['title'])} chars")
    if cur.get("hook"):
        st.write(f"**Hook:** {cur['hook']}")
    if cur.get("why"):
        st.write(f"**Why it works:** {cur['why']}")
    st.caption("Strength is Gemini's own estimate, not real YouTube data. "
               "Your real click-through rate after publishing is the true test.")

    b1, b2 = st.columns(2)
    if b1.button("✅ Use this", type="primary", use_container_width=True):
        record("used", cur)
        st.session_state.current = None
        st.rerun()
    if b2.button("⏭ Skip", use_container_width=True):
        record("skipped", cur)
        advance(format_name, topic)
        st.rerun()

if st.session_state.last_error and cur is not None:
    st.error(st.session_state.last_error)

st.divider()
with st.expander(f"Used titles ({len(st.session_state.history['used'])})"):
    for x in reversed(st.session_state.history["used"]):
        st.write(f"• {x['title']}  — {x.get('format', '')} · {x.get('score', '')}")
with st.expander(f"Skipped titles ({len(st.session_state.history['skipped'])})"):
    for x in reversed(st.session_state.history["skipped"][-50:]):
        st.write(f"• {x['title']}")

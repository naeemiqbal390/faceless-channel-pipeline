"""
chapters.py
Proposes YouTube chapters (timestamps for the description) from the scene
manifest's real timings. Plain Python, no Streamlit.

YouTube's rules for chapters: first timestamp is 0:00, at least 3 chapters,
each at least 10 seconds long, timestamps in ascending order.
"""

import re

# Where story beats usually fall (fraction of the video, default title), per chapter count.
# Matches the script prompt's act map; the 50% midpoint twist is always included.
BEATS = {
    3: [(0.0, "The Hook"), (0.40, "The Turn"), (0.80, "The Choice")],
    4: [(0.0, "The Hook"), (0.15, "Where It Started"), (0.50, "The Turn"), (0.80, "The Choice")],
    5: [(0.0, "The Hook"), (0.12, "Where It Started"), (0.30, "The First Trouble"), (0.50, "The Turn"), (0.80, "The Choice")],
    6: [(0.0, "The Hook"), (0.12, "Where It Started"), (0.30, "The First Trouble"), (0.50, "The Turn"),
        (0.65, "The Cost"), (0.80, "The Choice")],
    7: [(0.0, "The Hook"), (0.12, "Where It Started"), (0.30, "The First Trouble"), (0.50, "The Turn"),
        (0.65, "The Cost"), (0.80, "The Choice"), (0.93, "The Ending")],
}
TURN_WORDS = re.compile(
    r"^(but|then|until|suddenly|now|years|months|weeks|days|later|meanwhile|so|and then|that changed|everything)\b",
    re.I)
MIN_GAP = 60.0          # seconds between chapters we propose (YouTube minimum is 10)


def _to_sec(v):
    try:
        h, m, s = str(v).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def scene_timings(df):
    """df: manifest DataFrame (lower-cased columns). Returns ([(scene_id, start_sec)], total_seconds)."""
    rows, last_end = [], 0.0
    for i, row in df.reset_index(drop=True).iterrows():
        try:
            sid = int(row["scene_id"])
        except Exception:
            sid = i + 1
        start = _to_sec(row.get("start_time"))
        end = _to_sec(row.get("end_time"))
        if start is None:
            continue
        rows.append((sid, start))
        last_end = max(last_end, end if end is not None else start + 4.0)
    return rows, last_end


def format_timestamp(sec, long_video):
    sec = int(round(max(0, sec)))
    h, m, s = sec // 3600, sec // 60 % 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if long_video else f"{sec // 60}:{s:02d}"


def propose_chapters(starts, total_seconds, narration_by_scene=None):
    """
    starts: [(scene_id, start_sec)] in order. Returns [(start_sec, scene_index, default_title)].
    Boundaries follow the story's usual beats, then snap to a nearby scene that begins with a
    turn word ("But", "Then", ...) so a chapter starts where something actually changes.
    """
    if not starts or total_seconds <= 0:
        return []
    narration_by_scene = narration_by_scene or {}
    # how many chapters: about one per 2.5-3 minutes, between 4 and 7 (YouTube needs 3+)
    n = int(round(total_seconds / 170.0))
    n = max(4, min(7, n)) if total_seconds >= 150 else 3
    beats = BEATS[n]
    fractions = [f for f, _ in beats]
    titles = [t for _, t in beats]
    min_gap = min(MIN_GAP, max(10.0, total_seconds / (n * 1.6)))
    chosen = []
    for k, frac in enumerate(fractions):
        target = frac * total_seconds
        # nearest scene start to target
        idx = min(range(len(starts)), key=lambda i: abs(starts[i][1] - target))
        if k > 0:
            # snap to a turn word within +-6 scenes
            for off in sorted(range(-6, 7), key=abs):
                j = idx + off
                if 0 <= j < len(starts):
                    txt = (narration_by_scene.get(starts[j][0]) or "").strip()
                    if txt and TURN_WORDS.match(txt):
                        idx = j
                        break
        t = 0.0 if k == 0 else starts[idx][1]
        if chosen and t - chosen[-1][0] < min_gap:
            continue
        chosen.append((t, 0 if k == 0 else idx, titles[k]))
    return chosen


def section_excerpts(chapters, starts, narration_by_scene, words=70):
    """First ~70 words of narration in each chapter (to help an AI title them)."""
    out = []
    for k, (t, idx, _) in enumerate(chapters):
        end_idx = chapters[k + 1][1] if k + 1 < len(chapters) else len(starts)
        txt = " ".join((narration_by_scene.get(starts[i][0]) or "") for i in range(idx, min(end_idx, idx + 14)))
        out.append(" ".join(txt.split()[:words]))
    return out


def ai_titles(excerpts, video_title, api_key, gemini_json, model_override=""):
    """Teaser titles via Gemini; returns (titles_or_None, error_or_None)."""
    system = ("You write YouTube chapter titles for a narrated story video. For each section excerpt, write a "
              "short teaser title of 2 to 5 words. Never reveal twists or the ending. Plain words, Title Case, "
              "no emoji, no numbering, no quotes. Each title must be different. "
              'Return ONLY JSON: {"titles": [str, ...]} with exactly one title per excerpt, in order.')
    user = f"Video title: {video_title}\nSection excerpts:\n" + "\n".join(f"{i + 1}. {e}" for i, e in enumerate(excerpts))
    data, err = gemini_json(system, user, api_key, model_override)
    if not data:
        return None, err
    titles = data.get("titles") if isinstance(data, dict) else data
    if not isinstance(titles, list) or len(titles) != len(excerpts):
        return None, "Gemini returned the wrong number of titles."
    clean = [re.sub(r"\s+", " ", str(t)).strip().strip('"')[:60] for t in titles]
    if not all(clean):
        return None, "Gemini returned an empty title."
    return clean, None


LINE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s+(.+?)\s*$")


def parse_chapter_text(text):
    """Parses 'm:ss Title' lines. Returns ([(seconds, title)], [bad_line_texts])."""
    items, bad = [], []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        m = LINE.match(raw)
        if not m:
            bad.append(raw.strip()[:60])
            continue
        h = int(m.group(1) or 0)
        items.append((h * 3600 + int(m.group(2)) * 60 + int(m.group(3)), m.group(4)))
    return items, bad


def validate(items, total_seconds=None):
    """Plain-language problems against YouTube's chapter rules. Empty list = OK."""
    problems = []
    if len(items) < 3:
        problems.append("YouTube needs at least 3 chapters.")
    if items and items[0][0] != 0:
        problems.append("The first chapter must start at 0:00.")
    for a, b in zip(items, items[1:]):
        if b[0] <= a[0]:
            problems.append(f"Timestamps must go in order (check '{b[1][:30]}').")
        elif b[0] - a[0] < 10:
            problems.append(f"'{a[1][:30]}' is shorter than 10 seconds.")
    if total_seconds and items and total_seconds - items[-1][0] < 10:
        problems.append("The last chapter starts less than 10 seconds before the end.")
    return problems

"""
thumbnail_maker.py
Builds YouTube thumbnails (1280x720 JPEG, under 2 MB) from one of the
generated scene images plus bold headline text.

The text is drawn here with Pillow, NOT by the image model, so it is always
spelled correctly and always readable.

Three layouts are provided so you can upload up to three versions to
YouTube's "Test & compare":
  split_left   cream background, picture on the right, headline on the left
  split_right  cream background, picture on the left, headline on the right
  overlay      picture fills the frame, headline on a dark gradient
"""

import os
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
MAX_BYTES = 2 * 1024 * 1024

CREAM = (245, 241, 232)
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
RED = (214, 32, 32)

LAYOUTS = ["split_left", "split_right", "overlay"]

_BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font(size):
    for path in _BOLD_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def _text_size(draw, text, font, stroke=0):
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    return box[2] - box[0], box[3] - box[1]


def _wrap_words(draw, words, font, max_width, stroke):
    """Greedy wrap of (word, is_key) tuples into lines."""
    lines, current = [], []
    for item in words:
        trial = current + [item]
        text = " ".join(w for w, _ in trial)
        if current and _text_size(draw, text, font, stroke)[0] > max_width:
            lines.append(current)
            current = [item]
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _parse_headline(headline, keyword):
    key_words = {w.strip(".,!?:;\"'").upper() for w in (keyword or "").split()}
    words = []
    for w in (headline or "").split():
        words.append((w.upper(), w.strip(".,!?:;\"'").upper() in key_words))
    return words


def _fit_text(draw, words, box_w, box_h, stroke, max_lines=4):
    """Largest font size where the wrapped headline fits the box."""
    size = 220
    while size >= 36:
        font = _font(size)
        lines = _wrap_words(draw, words, font, box_w, stroke)
        line_h = int(size * 1.08)
        widest = max((_text_size(draw, " ".join(w for w, _ in ln), font, stroke)[0]
                      for ln in lines), default=0)
        if len(lines) <= max_lines and line_h * len(lines) <= box_h and widest <= box_w:
            return font, lines, line_h
        size -= 6
    font = _font(36)
    return font, _wrap_words(draw, words, font, box_w, stroke), int(36 * 1.08)


def _draw_headline(draw, words, box, text_color, key_color, stroke=0,
                   stroke_color=None, align="left"):
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    font, lines, line_h = _fit_text(draw, words, box_w, box_h, stroke)
    total_h = line_h * len(lines)
    y = y0 + (box_h - total_h) // 2
    for line in lines:
        line_text = " ".join(w for w, _ in line)
        line_w = _text_size(draw, line_text, font, stroke)[0]
        if align == "center":
            x = x0 + (box_w - line_w) // 2
        else:
            x = x0
        space_w = _text_size(draw, "A A", font)[0] - _text_size(draw, "AA", font)[0]
        for word, is_key in line:
            draw.text((x, y), word, font=font,
                      fill=key_color if is_key else text_color,
                      stroke_width=stroke, stroke_fill=stroke_color or BLACK)
            x += _text_size(draw, word, font, stroke)[0] + space_w
        y += line_h


def _cover(img, w, h):
    """Scale and center-crop img to exactly w x h."""
    scale = max(w / img.width, h / img.height)
    resized = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.LANCZOS)
    left = (resized.width - w) // 2
    top = (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


def make_thumbnail(image_path, headline, keyword, out_path, layout="split_left"):
    """
    Renders one thumbnail to out_path (JPEG). Returns out_path.

    headline : short text, ideally 2-4 words ("NO DENTIST?")
    keyword  : the word(s) of the headline to show in red ("DENTIST")
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Choose from {LAYOUTS}.")

    source = Image.open(image_path).convert("RGB")
    words = _parse_headline(headline, keyword)
    if not words:
        raise ValueError("Headline text is empty.")

    if layout == "overlay":
        canvas = _cover(source, WIDTH, HEIGHT).convert("RGBA")
        # dark gradient on the left 62% so white text stays readable
        gradient = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(gradient)
        fade_end = int(WIDTH * 0.62)
        for x in range(fade_end):
            alpha = int(215 * (1 - x / fade_end) ** 1.3)
            gdraw.line([(x, 0), (x, HEIGHT)], fill=(0, 0, 0, alpha))
        canvas = Image.alpha_composite(canvas, gradient).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        _draw_headline(draw, words, (60, 60, int(WIDTH * 0.58), HEIGHT - 60),
                       WHITE, RED, stroke=6, stroke_color=BLACK)
    else:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), CREAM)
        pic_w = 640  # picture is centre-cropped from the square; more room for text
        picture = _cover(source, pic_w, HEIGHT)
        draw = ImageDraw.Draw(canvas)
        if layout == "split_left":
            canvas.paste(picture, (WIDTH - pic_w, 0))
            _draw_headline(draw, words, (50, 40, WIDTH - pic_w - 30, HEIGHT - 40),
                           BLACK, RED, stroke=0)
        else:  # split_right
            canvas.paste(picture, (0, 0))
            _draw_headline(draw, words, (pic_w + 30, 40, WIDTH - 50, HEIGHT - 40),
                           BLACK, RED, stroke=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    quality = 92
    while True:
        canvas.save(out_path, "JPEG", quality=quality, optimize=True)
        if os.path.getsize(out_path) <= MAX_BYTES or quality <= 60:
            break
        quality -= 8
    return out_path

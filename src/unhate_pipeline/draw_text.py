from PIL import Image, ImageDraw, ImageFont


def _load_meme_font(size: int) -> ImageFont.ImageFont:
    # Try common bold fonts used for meme captions, then gracefully fall back.
    font_candidates = [
        "Impact.ttf",
        "impact.ttf",
        "Arial Bold.ttf",
        "arialbd.ttf",
        "DejaVuSans-Bold.ttf",
    ]

    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_line_to_width(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> list[str]:
    if not line:
        return [""]

    words = line.split()
    if len(words) <= 1:
        return [line]

    wrapped: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        left, top, right, bottom = draw.textbbox(
            (0, 0),
            candidate,
            font=font,
            stroke_width=stroke_width,
        )
        if (right - left) <= max_width:
            current = candidate
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


def draw_meme_text(
    image: Image.Image,
    text: str,
    center_xy: tuple[int, int],
    max_width_ratio: float = 0.90,
) -> Image.Image:
    """Draw meme-style caption text that is large, readable, and kept inside bounds."""
    if not text or not text.strip():
        return image

    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    cx = max(0, min(int(center_xy[0]), img_w))
    cy = max(0, min(int(center_xy[1]), img_h))
    max_text_width = max(40, int(img_w * max_width_ratio))

    # Size text from both image area and short side for robust scaling across formats.
    short_side = min(img_w, img_h)
    area_scale = (img_w * img_h) ** 0.5
    base_size = max(12, int(area_scale * 0.052))
    min_font_size = max(10, int(base_size * 0.60))
    max_font_size = max(min_font_size + 2, int(min(base_size * 1.45, short_side * 0.16)))

    source_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not source_lines:
        return image

    margin = max(8, short_side // 30)
    is_top_region = cy <= (img_h // 2)
    # Keep each caption block visually compact so it does not dominate the image.
    available_height = min((img_h // 2) - (2 * margin), int(img_h * 0.22))
    available_height = max(32, available_height)

    best_font = _load_meme_font(min_font_size)
    best_lines = source_lines
    best_stroke = max(1, min_font_size // 11)

    for size in range(max_font_size, min_font_size - 1, -2):
        font = _load_meme_font(size)
        stroke_width = max(1, size // 11)

        candidate_lines: list[str] = []
        for src in source_lines:
            candidate_lines.extend(
                _wrap_line_to_width(
                    draw=draw,
                    line=src,
                    font=font,
                    max_width=max_text_width,
                    stroke_width=stroke_width,
                )
            )

        fits = True
        for line in candidate_lines:
            left, top, right, bottom = draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=stroke_width,
            )
            if (right - left) > max_text_width:
                fits = False
                break

        if fits:
            ascent, descent = font.getmetrics()
            line_height = ascent + descent
            line_spacing = max(2, line_height // 8)
            block_height = (
                line_height * len(candidate_lines)
                + line_spacing * (len(candidate_lines) - 1)
            )
            if block_height > available_height:
                fits = False

        if fits:
            best_font = font
            best_lines = candidate_lines
            best_stroke = stroke_width
            break

    ascent, descent = best_font.getmetrics()
    line_height = ascent + descent
    line_spacing = max(2, line_height // 8)
    block_height = line_height * len(best_lines) + line_spacing * (len(best_lines) - 1)

    if is_top_region:
        y = margin
    else:
        y = img_h - margin - block_height

    y = max(margin, min(y, img_h - block_height - margin))

    for line in best_lines:
        left, top, right, bottom = draw.textbbox(
            (0, 0),
            line,
            font=best_font,
            stroke_width=best_stroke,
        )
        line_w = right - left

        x = cx - (line_w // 2)
        x = max(margin, min(x, img_w - line_w - margin))

        draw.text(
            (x, y),
            line,
            font=best_font,
            fill=(255, 255, 255),
            stroke_width=best_stroke,
            stroke_fill=(0, 0, 0),
        )
        y += line_height + line_spacing

    return image
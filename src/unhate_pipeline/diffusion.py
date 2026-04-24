import sys
import torch
from PIL import Image

from diffusers import Flux2KleinPipeline
from draw_text import draw_meme_text
from prompt import ERASE_TEXT_PROMPT


def instantiate_diffusion(model_name: str, cache_dir: str | None =None):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[info] Using dtype: {torch_dtype}", file=sys.stderr)
    print(f"[info] Loading diffusion model from repo: {model_name}", file=sys.stderr)

    pipe = Flux2KleinPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    pipe.enable_sequential_cpu_offload()
    return pipe


@torch.inference_mode()
def run_diffusion(pipe, image, prompt, generator=None, sigmas=None) -> Image.Image:
    w, h = image.size
    h = min(round(h / 64) * 64, 1024)
    w = min(round(w / 64) * 64, 1024)
    
    # image.resize((1024, 1024))
    return pipe(
        prompt=prompt,
        image=image,
        height=h,
        width=w,
        guidance_scale=1.0,
        num_inference_steps=6,
        sigmas=torch.linspace(sigmas, 1e-5, 4).tolist() if sigmas is not None else None,
        generator=generator
    ).images[0]


def erase_text(pipe, image: Image.Image, generator=None) -> Image.Image:
    return run_diffusion(pipe, image, ERASE_TEXT_PROMPT, generator=generator)


def detect_text_position(mitigation: dict, image_size: tuple) -> dict:
    """
    Infer text position from the meme format and original_text content.
    Returns pixel coordinates for top and/or bottom text.
    Falls back to classic meme layout if format is unknown.
    """
    w, h = image_size
    positions = {}

    original = mitigation.get("original_text") or ""

    top_y = max(20, int(h * 0.06))
    bottom_y = h - max(20, int(h * 0.06))

    lines = [l.strip() for l in original.split("\n") if l.strip()]
    if len(lines) >= 2:
        positions["top"] = (w // 2, top_y)
        positions["bottom"] = (w // 2, bottom_y)
    elif len(lines) == 1:
        positions["bottom"] = (w // 2, bottom_y)

    return positions


def handle_text_mitigation(
    pipe,
    image: Image.Image,
    mitigation: dict,
    generator=None
) -> Image.Image:
    """
    Full pipeline for text-based hate mitigation:
    1. Erases original text
    2. Draw replacement text if provided, preserving meme format and style (e.g., top/bottom text layout, font style, and position).
    """
    replacement = mitigation.get("replacement_text")
    hate_loc = mitigation["hate_location"]

    if hate_loc not in ("TEXT_ONLY", "COMBINED", "INTERSECTIONAL"):
        return image

    # Remove text
    print("[info] Removing text...")
    image = erase_text(pipe, image, generator=generator)

    # Rewrite with replacement text if provided
    replacement_text = "" if replacement is None else str(replacement)
    replacement_text = replacement_text.replace("\\n", "\n")

    print(f"[info] Writing replacement text: '{replacement_text}'...")
    positions = detect_text_position(mitigation, image.size)

    # Split only on the first newline; collapse any remaining newlines in the second chunk.
    normalized_replacement = replacement_text.strip()
    if "\n" in normalized_replacement:
        first_line, remainder = normalized_replacement.split("\n", 1)
        remainder = remainder.replace("\n", "")
        lines = [part.strip() for part in (first_line, remainder) if part.strip()]
    else:
        lines = [normalized_replacement] if normalized_replacement else []

    img_w, img_h = image.size
    top_pos = positions.get("top", (img_w // 2, max(20, int(img_h * 0.06))))
    bottom_pos = positions.get("bottom", (img_w // 2, img_h - max(20, int(img_h * 0.06))))

    if len(lines) >= 2:
        image = draw_meme_text(image, lines[0], top_pos)
        image = draw_meme_text(image, lines[-1], bottom_pos)
    elif lines:
        image = draw_meme_text(image, lines[0], bottom_pos)
    elif replacement_text.strip():
        image = draw_meme_text(image, replacement_text.strip(), bottom_pos)

    return image


def mitigate_image(pipe, image: Image.Image, mitigation: dict, generator=None) -> Image.Image:
    hate_loc = mitigation["hate_location"]

    # A: Handle visual mitigation with diffusion model (if needed)
    if hate_loc in ("VISUAL_ONLY", "COMBINED", "INTERSECTIONAL", "STRUCTURAL"):
        print(f"[info] Mitigating visual elements...")
        image = run_diffusion(pipe, image, mitigation["flux_prompt"], generator=generator)

    # B: Handle text mitigation (Remove/replace hateful text) if needed
    if hate_loc in ("TEXT_ONLY", "COMBINED", "INTERSECTIONAL") and mitigation.get("replacement_text"):
        image = handle_text_mitigation(pipe, image, mitigation, generator=generator)

    return image

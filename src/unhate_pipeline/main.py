import sys
import argparse

import torch
from pathlib import Path
from diffusers.utils import load_image

from diffusion import instantiate_diffusion, mitigate_image
from utils import parse_prompt_generation, parse_hateful_response
from vlm import (
    instantiate_vlm,
    load_cls_head,
    detect_hateful_meme,
    detect_hateful_meme_cls_head,
    get_diffusion_prompt,
)


def run_pipeline(vlm, vlm_processor, diffusion_model, image_path, cls_head=None):
    image = load_image(str(image_path))
    mitigated_dir = image_path.parent / "mitigated"
    mitigated_dir.mkdir(exist_ok=True)

    # ── Detection ─────────────────────────────────────────────────────────────
    # Two modes depending on whether a classification head was loaded:
    #
    #   cls_head provided → forward pass only, no generation.
    #     Fast, deterministic, no risk of malformed JSON output.
    #     Returns (is_hateful: bool, probability: float).
    #
    #   no cls_head → VLM generates a JSON response (original behaviour).
    #     Slower, but also produces a description used for logging.
    #
    if cls_head is not None:
        is_hateful, probability = detect_hateful_meme_cls_head(
            vlm, vlm_processor, cls_head, image_path
        )
        print(f"\nDetection (cls head): is_hateful={is_hateful}, "
              f"probability={probability:.3f}\n")
    else:
        hateful_response = detect_hateful_meme(vlm, vlm_processor, image_path)
        print(f"\nHateful detection output:\n{hateful_response}\n")
        is_hateful, probability, _ = parse_hateful_response(hateful_response)

    if probability < 0.2:
        print("The meme is not hateful — passing through unchanged.")
        image.save(mitigated_dir / f"{image_path.stem}_mitigated.png")
        return

    # ── Mitigation prompt generation ──────────────────────────────────────────
    # Always uses the VLM generatively, regardless of detection mode.
    # The classification head is only for the binary detection decision;
    # generating a structured diffusion plan always requires generation.
    print("[info] Generating diffusion prompt...", file=sys.stderr)
    diffusion_prompt = get_diffusion_prompt(vlm, vlm_processor, image_path)
    print(f"\nGenerated diffusion prompt:\n{diffusion_prompt}\n")
    mitigation = parse_prompt_generation(diffusion_prompt)

    # ── Image mitigation ──────────────────────────────────────────────────────
    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(42)
    mitigated_image = mitigate_image(diffusion_model, image, mitigation, generator=generator)
    mitigated_image.save(mitigated_dir / f"{image_path.stem}_mitigated.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_name", default="google/gemma-4-31b-it")
    parser.add_argument("--diffusion_model_name", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--data_path", default="data/")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--adapter_path", default=None,
                        help="LoRA adapter directory (checkpoints/detect/adapter_detect). "
                             "Mutually exclusive with --cls_head_path.")
    parser.add_argument("--cls_head_path", default=None,
                        help="Path to classifier.pt produced by train_cls_head.py "
                             "(e.g. checkpoints/cls_head/cls_head/classifier.pt). "
                             "When provided, detection uses a forward pass instead of "
                             "generation. Generation for the diffusion prompt is unaffected.")
    args = parser.parse_args()

    if args.adapter_path and args.cls_head_path:
        print("[error] --adapter_path and --cls_head_path are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[info] VLM: {args.vlm_name}", file=sys.stderr)
    print(f"[info] Data path: {args.data_path}", file=sys.stderr)

    vlm, vlm_processor = instantiate_vlm(
        args.vlm_name, args.cache_dir, args.adapter_path
    )

    cls_head = None
    if args.cls_head_path:
        print(f"[info] Loading classification head from {args.cls_head_path}",
              file=sys.stderr)
        cls_head = load_cls_head(vlm, args.cls_head_path)

    print(f"[info] Loading diffusion model: {args.diffusion_model_name}", file=sys.stderr)
    diffusion_model = instantiate_diffusion(
        args.diffusion_model_name, cache_dir=args.cache_dir
    )

    image_paths = sorted(Path(args.data_path).glob("*.png"))
    if not image_paths:
        print(f"[error] No .png images found in: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Found {len(image_paths)} image(s).", file=sys.stderr)

    for image_path in image_paths:
        print(f"\n[info] Processing image: {image_path}", file=sys.stderr)
        run_pipeline(vlm, vlm_processor, diffusion_model, image_path, cls_head=cls_head)


if __name__ == "__main__":
    main()
  
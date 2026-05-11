import sys
import argparse

import torch
from pathlib import Path
from diffusers.utils import load_image

from diffusion import instantiate_diffusion, mitigate_image
from utils import (
    parse_prompt_generation,
    parse_hateful_response,
    parse_affect_response,
)
from vlm import (
    instantiate_vlm,
    get_diffusion_prompt,
    classify_meme_affect,
    detect_hateful_meme_affect_aware,
)


def run_pipeline(vlm, vlm_processor, diffusion_model, image_path):
    # load image
    image = load_image(str(image_path))
    mitigated_dir = image_path.parent / "mitigated"
    mitigated_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: classify affective / communicative properties of the meme
    print("[info] Classifying meme affect...", file=sys.stderr)
    affect_response = classify_meme_affect(vlm, vlm_processor, image_path)
    print(f"\nAffect classification output:\n{affect_response}\n")

    affect = parse_affect_response(affect_response)
    print(f"[info] Parsed affect: {affect}", file=sys.stderr)

    # Step 2: detect hatefulness using affect-aware prompting
    print("[info] Running affect-aware hateful detection...", file=sys.stderr)
    hateful_response = detect_hateful_meme_affect_aware(
        vlm,
        vlm_processor,
        image_path,
        affect,
    )

    print(f"\nAffect-aware hateful detection output:\n{hateful_response}\n")
    is_hateful, probability, description = parse_hateful_response(hateful_response)

    print(f"[info] Hateful classification: {is_hateful}", file=sys.stderr)
    print(f"[info] Hateful probability: {probability}", file=sys.stderr)
    print(f"[info] Hateful description: {description}", file=sys.stderr)

    # Step 3: keep the existing threshold logic
    if probability < 0.2:
        print("The meme is not hateful.")
        mitigated_output_path = mitigated_dir / f"{image_path.stem}_mitigated.png"
        image.save(mitigated_output_path)
        return

    # Step 4: keep the existing mitigation prompt generation
    print(f"[info] Generating prompt for diffusion model...", file=sys.stderr)
    diffusion_prompt = get_diffusion_prompt(vlm, vlm_processor, image_path)
    print(f"\nGenerated diffusion prompt:\n{diffusion_prompt}\n")

    mitigation = parse_prompt_generation(diffusion_prompt)

    # Step 5: keep the existing mitigation process
    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(42)

    mitigated_image = mitigate_image(
        diffusion_model,
        image,
        mitigation,
        generator=generator,
    )

    mitigated_image.save(mitigated_dir / f"{image_path.stem}_mitigated.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_name", required=False, default="Qwen/Qwen3.6-27B")
    parser.add_argument("--diffusion_model_name", required=False, default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--data_path", required=False, default="data/")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    print(f"[info] Starting inference with model: {args.vlm_name}", file=sys.stderr)
    print(f"[info] Data path: {args.data_path}", file=sys.stderr)

    vlm, vlm_processor = instantiate_vlm(args.vlm_name, args.cache_dir)
    print(f"[info] Model loaded on device: {vlm.device}", file=sys.stderr)

    print(f"[info] Loading diffusion model: {args.diffusion_model_name}", file=sys.stderr)
    diffusion_model = instantiate_diffusion(args.diffusion_model_name, cache_dir=args.cache_dir)

    image_paths = sorted(Path(args.data_path).glob("*.png"))
    if not image_paths:
        print(f"[error] No .png images found in: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Found {len(image_paths)} image(s).", file=sys.stderr)

    for image_path in image_paths:
        print(f"\n[info] Processing image: {image_path}", file=sys.stderr)
        run_pipeline(vlm, vlm_processor, diffusion_model, image_path)


if __name__ == "__main__":
    main()
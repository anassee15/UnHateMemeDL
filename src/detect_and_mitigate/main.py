import argparse
import sys

from pathlib import Path
from diffusers.utils import load_image

from diffusion import instantiate_diffusion, remove_text_from_image, add_text_to_image, replace_text_in_image, unhate_image
from utils import parse_hateful_response, parse_hate_type_response, parse_hate_source_response
from vlm import instantiate_vlm, detect_hateful_meme, detect_hate_modality, detect_hate_type, mitigate_hateful_text


def run_pipeline(vlm, vlm_processor, diffusion_model, image_path):
    # hateful detection
    print(f"[info] Running hateful detection for image...", file=sys.stderr)
    hateful_response = detect_hateful_meme(vlm, vlm_processor, image_path)
    print(f"\nHateful detection output:\n{hateful_response}\n")
    is_hateful, probability, description = parse_hateful_response(hateful_response)

    if not is_hateful:
        print("The meme is not hateful.")
        return

    # type of hate
    print(f"[info] Running hate type classification for image...", file=sys.stderr)
    hate_type_response = detect_hate_type(vlm, vlm_processor, image_path)
    print(f"\nHate type output:\n{hate_type_response}\n")
    hate_type = parse_hate_type_response(hate_type_response)
    print(f"\nHate type: {hate_type}")
    
    if hate_type == "unimodal-hate":
        # source of hate
        print(f"[info] Running hate modality classification...", file=sys.stderr)
        hate_source_response = detect_hate_modality(vlm, vlm_processor, image_path)
        hate_source = parse_hate_source_response(hate_source_response)
        print(f"\nHate source: {hate_source}")

        image = load_image(image_path)
        
        if hate_source == "hate from text":
            print(f"[info] Running text mitigation...", file=sys.stderr)
            mitigated_text = mitigate_hateful_text(vlm, vlm_processor, image_path)
            print(f"\nMitigated text:\n{mitigated_text}\n")
            print(f"[info] Removing text from image...", file=sys.stderr)
            removed_text_image = replace_text_in_image(diffusion_model, image, new_text=mitigated_text)
            removed_text_image.save(image_path.parent / f"{image_path.stem}_mitigated.png")

        elif hate_source == "hate from image":
            print(f"[info] Running image mitigation...", file=sys.stderr)
            mitigated_image = unhate_image(diffusion_model, image, details_of_hate=hate_type)
            mitigated_image.save(image_path.parent / f"{image_path.stem}_mitigated.png")

    elif hate_type == "multimodal-hate":
        print(f"[info] Running text mitigation...", file=sys.stderr)
        mitigated_text = mitigate_hateful_text(vlm, vlm_processor, image_path)
        print(f"\nMitigated text:\n{mitigated_text}\n")
        print(f"[info] Removing text from image...", file=sys.stderr)
        removed_text_image = remove_text_from_image(diffusion_model, image, text_to_remove=mitigated_text)
        print(f"[info] Running image mitigation...", file=sys.stderr)
        unhated_image = unhate_image(diffusion_model, removed_text_image, details_of_hate=hate_type)
        mitigated_image = add_text_to_image(diffusion_model, unhated_image, new_text=mitigated_text)
        mitigated_image.save(image_path.parent / f"{image_path.stem}_mitigated.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_name", required=False, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--diffusion_model_name", default="black-forest-labs/FLUX.2-klein-9B")
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
  
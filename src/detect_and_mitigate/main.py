import argparse
import sys

import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

from vlm import detect_hateful_meme, detect_hate_modality, detect_hate_type, mitigate_hateful_text
from utils import parse_hateful_response, parse_hate_type_response, parse_hate_source_response


def instantiate_model(model_name: str, cache_dir: str | None = None) -> AutoModelForImageTextToText:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[info] Using dtype: {dtype}", file=sys.stderr)
    print("[info] Loading processor...", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    print("[info] Loading model...", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
    )
    return model, processor


def run_pipeline(model, processor, image_path, cache_dir=None):
    # hateful detection
    print(f"[info] Running hateful detection for image...", file=sys.stderr)
    hateful_response = detect_hateful_meme(model, processor, image_path)
    print(f"\nHateful detection output:\n{hateful_response}\n")
    is_hateful, probability = parse_hateful_response(hateful_response)

    if not is_hateful:
        print("The meme is not hateful.")
        return

    # type of hate
    print(f"[info] Running hate type classification for image...", file=sys.stderr)
    hate_type_response = detect_hate_type(model, processor, image_path)
    print(f"\nHate type output:\n{hate_type_response}\n")

    hate_type = parse_hate_type_response(hate_type_response)
    print(f"\nHate type: {hate_type}")

    # source of hate
    print(f"[info] Running hate modality classification...", file=sys.stderr)
    hate_source_response = detect_hate_modality(model, processor, image_path)
    hate_source = parse_hate_source_response(hate_source_response)
    print(f"\nHate source: {hate_source}")
    
    if hate_source in ["hate from text", "hate from both"]:
        print(f"[info] Running text mitigation...", file=sys.stderr)
        mitigated_text = mitigate_hateful_text(model, processor, image_path)
        print(f"\nMitigated text:\n{mitigated_text}\n")
    
    if hate_source in ["hate from image", "hate from both"]:
        print(f"[info] Running image mitigation...", file=sys.stderr)
        # TODO: Image mitigation

    # TODO: Finalisation (merge of mitigated text and image, saving results, etc.)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=False, default="data/")
    parser.add_argument("--cache_dir", default=None)

    args = parser.parse_args()
    print(f"[info] Starting inference with model: {args.model_name}", file=sys.stderr)
    print(f"[info] Data path: {args.data_path}", file=sys.stderr)

    model, processor = instantiate_model(args.model_name, args.cache_dir)
    print(f"[info] Model loaded on device: {model.device}", file=sys.stderr)

    image_paths = sorted(Path(args.data_path).glob("*.png"))
    if not image_paths:
        print(f"[error] No .png images found in: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Found {len(image_paths)} image(s).", file=sys.stderr)

    for image_path in image_paths:
        print(f"\n[info] Processing image: {image_path}", file=sys.stderr)
        run_pipeline(model, processor, image_path, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
  
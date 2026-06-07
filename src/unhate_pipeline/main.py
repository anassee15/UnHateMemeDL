import sys
import gc
import json
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

# Below this probability the meme is treated as non-hateful and passed through.
PASSTHROUGH_THRESHOLD = 0.5


def plan_image(vlm, vlm_processor, image_path, cls_head=None, thinking=True):
    """Detect the meme and, if hateful, generate its mitigation plan with the VLM."""
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

    if probability < PASSTHROUGH_THRESHOLD:
        return {"image_path": str(image_path), "probability": probability, "mitigation": None}

    print("[info] Generating diffusion prompt...", file=sys.stderr)
    diffusion_prompt = get_diffusion_prompt(vlm, vlm_processor, image_path, thinking=thinking)
    print(f"\nGenerated diffusion prompt:\n{diffusion_prompt}\n")
    mitigation = parse_prompt_generation(diffusion_prompt)
    return {"image_path": str(image_path), "probability": probability, "mitigation": mitigation}


def release_vlm(vlm, cls_head=None):
    """Free the VLM (and head) so the diffusion model can use the full GPU."""
    del vlm
    if cls_head is not None:
        del cls_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_mitigation(diffusion_model, plan):
    image_path = Path(plan["image_path"])
    mitigated_dir = image_path.parent / "mitigated"
    mitigated_dir.mkdir(exist_ok=True)
    out_path = mitigated_dir / f"{image_path.stem}_mitigated.png"

    image = load_image(str(image_path))
    if plan["mitigation"] is None:
        print(f"[info] {image_path.name}: not hateful — passing through unchanged.")
        image.save(out_path)
        return

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(42)
    mitigated_image = mitigate_image(diffusion_model, image, plan["mitigation"], generator=generator)
    mitigated_image.save(out_path)


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
    parser.add_argument("--mitigation_adapter", default=None,
                        help="LoRA adapter directory produced by train_mitigation.py "
                             "(e.g. checkpoints/mitigation/adapter_mitigation).")
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable the step-by-step CoT reasoning block in the "
                             "mitigation prompt. On by default.")
    parser.add_argument("--diffusion_offload", action="store_true",
                        help="Use sequential CPU offload for the diffusion model instead "
                             "of loading it fully on CUDA. Slower, for low-VRAM GPUs.")
    args = parser.parse_args()

    if args.adapter_path and args.cls_head_path:
        print("[error] --adapter_path and --cls_head_path are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[info] VLM: {args.vlm_name}", file=sys.stderr)
    print(f"[info] Data path: {args.data_path}", file=sys.stderr)

    image_paths = sorted(Path(args.data_path).glob("*.png"))
    if not image_paths:
        print(f"[error] No .png images found in: {args.data_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[info] Found {len(image_paths)} image(s).", file=sys.stderr)

    # Phase 1: run all VLM work (detection + mitigation plans), then release the VLM.
    vlm, vlm_processor = instantiate_vlm(
        args.vlm_name, args.cache_dir, args.adapter_path,
        mitigation_adapter_path=args.mitigation_adapter,
    )
    cls_head = None
    if args.cls_head_path:
        print(f"[info] Loading classification head from {args.cls_head_path}", file=sys.stderr)
        cls_head = load_cls_head(vlm, args.cls_head_path)

    plans_dir = Path(args.data_path) / "plans"
    plans_dir.mkdir(exist_ok=True)
    plans = []
    for image_path in image_paths:
        print(f"\n[info] [VLM] Planning: {image_path}", file=sys.stderr)
        plan = plan_image(vlm, vlm_processor, image_path,
                          cls_head=cls_head, thinking=not args.no_thinking)
        (plans_dir / f"{image_path.stem}.json").write_text(json.dumps(plan, indent=2))
        plans.append(plan)

    release_vlm(vlm, cls_head)
    print("[info] VLM released; GPU memory reclaimed.", file=sys.stderr)

    # Phase 2: load the diffusion model with the full GPU and apply the saved plans.
    print(f"[info] Loading diffusion model: {args.diffusion_model_name}", file=sys.stderr)
    diffusion_model = instantiate_diffusion(
        args.diffusion_model_name, cache_dir=args.cache_dir, offload=args.diffusion_offload,
    )
    for plan in plans:
        print(f"\n[info] [Diffusion] Mitigating: {plan['image_path']}", file=sys.stderr)
        apply_mitigation(diffusion_model, plan)


if __name__ == "__main__":
    main()

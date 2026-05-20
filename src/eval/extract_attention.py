#!/usr/bin/env python3
"""
Saliency map heatmap extraction via input gradients (vanilla gradient method).

Selects one image from each confusion-matrix cell (TP, FP, TN, FN) from
report/categorized_v2.csv, runs both the 'baseline' and 'categorized_v2'
detection pipelines for each image, computes input-gradient saliency maps,
and writes JSON files to report/heatmaps/{pipeline}/{id}.json.

The saliency map measures how much each pixel in the input image influences
the model's hate-detection response, computed via d(NLL_loss)/d(pixel_values).

Usage:
    python3 src/eval/extract_attention.py \\
        --img_dir data/eval_data \\
        --vlm_name Qwen/Qwen2.5-VL-7B-Instruct \\
        --cache_dir /scratch/hf_cache
"""

import sys
import json
import argparse
import csv
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

from vlm import instantiate_vlm, run_vlm, _parse_category
from utils import parse_hateful_response
from affect_prompting import build_categorized_v2_detection_prompt, MEME_CATEGORY_PROMPT
from prompt import HATEFUL_DETECTION_PROMPT


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def load_cases(csv_path: Path) -> dict:
    """
    Scan categorized_v2.csv and return one CSV row per confusion-matrix cell:
      TP  label_true=1, label_pred=1
      FP  label_true=0, label_pred=1
      TN  label_true=0, label_pred=0
      FN  label_true=1, label_pred=0
    """
    targets = {"TP": (1, 1), "FP": (0, 1), "TN": (0, 0), "FN": (1, 0)}
    remaining = dict(targets)
    cases: dict = {}

    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            try:
                lt = int(row["label_true"])
                lp = int(row["label_pred"])
            except (KeyError, ValueError):
                continue
            for case, (rt, rp) in list(remaining.items()):
                if lt == rt and lp == rp:
                    cases[case] = row
                    del remaining[case]
                    break
            if not remaining:
                break

    for case in targets:
        if case not in cases:
            print(f"[warn] No CSV row found for case {case}", file=sys.stderr)

    return cases


# ---------------------------------------------------------------------------
# Saliency map via input gradients (vanilla gradient / input × gradient)
# ---------------------------------------------------------------------------

def compute_saliency_map(model, processor, image_path, prompt,
                         max_new_tokens=512, temperature=0.95,
                         thinking=False, system_prompt=""):
    """
    Compute a pixel-level saliency map using the vanilla input gradient method.

    Steps:
      1. Build inputs and run model.generate() to obtain the text response.
      2. Run a teacher-forced forward pass with requires_grad=True on pixel_values.
         Labels are set to -100 for prompt tokens (ignored) and the generated
         token IDs for the response tokens — so the NLL loss scores only the
         model's hate-detection answer.
      3. Backpropagate the loss to obtain d(loss)/d(pixel_values).
      4. Take |grad|, average across color channels → 2D saliency map [H, W].
      5. Normalize to [0, 1].

    Returns:
        (decoded_text: str, saliency: np.ndarray[H, W] | None)
        saliency is None if gradients could not be computed.
    """
    image = Image.open(image_path).convert("RGB")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    })
    text_template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )

    inputs = processor(text=[text_template], images=[image], return_tensors="pt")

    # Move all tensors to the device of the first model parameter.
    # With device_map="auto" this is typically cuda:0 (the vision encoder device).
    param_device = next(model.parameters()).device
    inputs = {k: v.to(param_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Step 1: generate the full text response (no gradients needed here)
    # ------------------------------------------------------------------
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    # Clone to convert the inference-mode tensor into a normal autograd-compatible
    # tensor before using it in the teacher-forced backward pass.
    generated = generated.clone()

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        generated[:, prompt_len:], skip_special_tokens=True
    )[0]

    # ------------------------------------------------------------------
    # Step 2: forward pass with gradient on pixel_values
    # ------------------------------------------------------------------
    # Feeding the full generated sequence back as input_ids triggers a shape
    # mismatch in Qwen2.5-VL's 3D-RoPE get_rope_index: the model builds
    # input_token_type from the original prompt-length structure, but our
    # all-ones mask covers prompt + generated tokens.  Fix: use the original
    # prompt inputs unchanged and compute NLL loss on the model's prediction
    # of the first generated token from the last prompt position.  This is
    # architecturally equivalent for saliency purposes.

    # Cast pixel_values to float32 so gradients accumulate at full precision
    # even when the model computes in bfloat16.
    pixel_values = inputs["pixel_values"].float().detach().requires_grad_(True)

    fwd_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": pixel_values,
    }
    for k, v in inputs.items():
        if k not in ("input_ids", "attention_mask", "pixel_values"):
            fwd_inputs[k] = v

    first_gen_id = generated[:, prompt_len].clone()  # [1] — token to predict

    model.zero_grad()
    torch.cuda.empty_cache()
    try:
        with torch.enable_grad():
            outputs = model(**fwd_inputs)
            logit_last = outputs.logits[:, -1, :]  # prediction after the prompt
            loss = torch.nn.functional.cross_entropy(logit_last, first_gen_id)
            loss.backward()
    except Exception as exc:
        print(f"    [warn] Gradient computation failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return decoded, None

    if pixel_values.grad is None:
        print(
            "    [warn] pixel_values.grad is None — gradient did not flow through "
            "the vision encoder. This can happen if the model uses non-differentiable "
            "operations (e.g. discrete token sampling, in-place ops, or certain "
            "custom attention kernels).",
            file=sys.stderr,
        )
        return decoded, None

    # ------------------------------------------------------------------
    # Step 3: aggregate to 2D saliency map
    # ------------------------------------------------------------------
    grad = pixel_values.grad.detach().abs()

    if grad.dim() == 4:
        # Standard [batch, C, H, W] layout (e.g. LLaVA, InstructBLIP)
        saliency = grad.squeeze(0).mean(dim=0)        # [H, W]
    elif grad.dim() == 2 and "image_grid_thw" in inputs:
        # Qwen2.5-VL: pixel_values is [N_patches, C*patch_h*patch_w]
        # Use image_grid_thw = [t, h, w] to reconstruct the 2D patch grid.
        thw = inputs["image_grid_thw"][0]
        n_t, n_h, n_w = int(thw[0]), int(thw[1]), int(thw[2])
        patch_sal = grad.mean(dim=1)                  # [N_patches]
        saliency = patch_sal[: n_t * n_h * n_w].reshape(n_t * n_h, n_w)
    else:
        saliency = grad.mean(dim=0).unsqueeze(0)      # graceful fallback

    saliency = saliency.cpu().float().numpy()

    lo, hi = float(saliency.min()), float(saliency.max())
    print(
        f"    [info] raw gradient: min={lo:.6f}  max={hi:.6f}  range={hi - lo:.6f}",
        file=sys.stderr,
    )

    if hi > lo:
        saliency = (saliency - lo) / (hi - lo)
    else:
        saliency[:] = 0.0

    return decoded, saliency


def saliency_to_grid(saliency_2d: np.ndarray) -> list[list[float]]:
    """Convert a 2D numpy saliency array to a JSON-serializable list-of-lists."""
    return [[float(v) for v in row] for row in saliency_2d]


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------

def run_pipeline_with_saliency(model, processor, img_path, pipeline,
                                max_new_tokens=512, temperature=0.95):
    """
    Run 'baseline' or 'categorized_v2' hate detection and compute the saliency map.

    Returns (prob_pred: float, saliency_2d: np.ndarray | None).
    """
    if pipeline == "baseline":
        print("    [baseline] hate detection + saliency", file=sys.stderr)
        raw, saliency = compute_saliency_map(
            model, processor, img_path, HATEFUL_DETECTION_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    elif pipeline == "categorized_v2":
        print("    [categorized_v2] step 1 — category", file=sys.stderr)
        cat_raw = run_vlm(
            model, processor, img_path, MEME_CATEGORY_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        category = _parse_category(cat_raw)
        print(f"    [categorized_v2] category={category}", file=sys.stderr)

        print("    [categorized_v2] step 2 — hate detection + saliency", file=sys.stderr)
        hate_prompt = build_categorized_v2_detection_prompt(category)
        raw, saliency = compute_saliency_map(
            model, processor, img_path, hate_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}")

    try:
        _, prob, _ = parse_hateful_response(raw)
    except Exception as exc:
        print(f"    [warn] Could not parse probability: {exc}", file=sys.stderr)
        prob = 0.0

    return prob, saliency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SMOKE_TEST_ID = "57823"  # FP case used as smoke test


def main():
    parser = argparse.ArgumentParser(
        description="Extract pixel-level saliency heatmaps via input gradients"
    )
    parser.add_argument("--img_dir", required=True,
                        help="Dataset root; image paths from the CSV are resolved as <img_dir>/<img>")
    parser.add_argument("--vlm_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--csv", default="report/categorized_v2.csv",
                        help="Path to categorized_v2 results CSV")
    parser.add_argument("--out_dir", default="report/heatmaps",
                        help="Output root; JSONs written to <out_dir>/<pipeline>/<id>.json")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_smoke_test", action="store_true",
                        help="Skip the smoke test and run all cases immediately")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    img_root = Path(args.img_dir)
    out_root = Path(args.out_dir)

    # 1. Select one example per confusion-matrix cell
    print(f"[info] Loading cases from {csv_path}", file=sys.stderr)
    cases = load_cases(csv_path)
    if not cases:
        print("[error] No valid cases found in CSV.", file=sys.stderr)
        sys.exit(1)

    print("[info] Selected examples:", file=sys.stderr)
    for case, row in sorted(cases.items()):
        print(f"  {case}: id={row['id']}  img={row['img']}", file=sys.stderr)

    # 2. Load model
    model, processor = instantiate_vlm(args.vlm_name, args.cache_dir)

    # 3. Smoke test on the FP case (id=57823) before running everything
    if not args.skip_smoke_test:
        smoke_row = next(
            (r for r in cases.values() if r["id"] == SMOKE_TEST_ID),
            cases.get("FP"),  # fall back to whichever row is the FP case
        )
        if smoke_row is None:
            print("[warn] Smoke test case not found — skipping smoke test", file=sys.stderr)
        else:
            smoke_id = smoke_row["id"]
            smoke_path = img_root / smoke_row["img"]
            print(f"\n[smoke test] id={smoke_id}  file={smoke_path.name}", file=sys.stderr)

            if not smoke_path.exists():
                print(f"[error] Smoke test image not found: {smoke_path}", file=sys.stderr)
                sys.exit(1)

            try:
                smoke_prob, smoke_saliency = run_pipeline_with_saliency(
                    model, processor, smoke_path, "baseline",
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:
                print(f"[error] Smoke test failed: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.exit(1)

            if smoke_saliency is None:
                print(
                    "\n[STOP] Smoke test: gradient did not flow — saliency is None.\n"
                    "Possible causes:\n"
                    "  1. The model uses non-differentiable ops in the vision encoder\n"
                    "     (e.g. custom CUDA kernels without autograd support).\n"
                    "  2. torch.inference_mode() was not fully exited before the grad pass.\n"
                    "  3. The pixel_values tensor was not in the computation graph\n"
                    "     (e.g. copied to a new tensor without grad before forward pass).\n"
                    "Stopping. Inspect the traceback above for details.",
                    file=sys.stderr,
                )
                sys.exit(1)

            H, W = smoke_saliency.shape
            sal_range = float(smoke_saliency.max() - smoke_saliency.min())
            print(
                f"[smoke test] grid shape: {H}×{W}  normalized range: {sal_range:.4f}",
                file=sys.stderr,
            )

            if sal_range < 0.05:
                print(
                    "\n[STOP] Smoke test: saliency map is nearly flat "
                    f"(max - min = {sal_range:.4f} < 0.05).\n"
                    "The gradient is technically flowing but carrying no meaningful\n"
                    "spatial signal. Likely causes:\n"
                    "  1. The model computes in bfloat16 internally, causing gradient\n"
                    "     underflow when backpropagating through many layers.\n"
                    "  2. The NLL loss is averaged over 500+ tokens; pixel gradients\n"
                    "     are dominated by text-token contributions and washed out.\n"
                    "  3. The vision encoder uses operations that block gradient flow\n"
                    "     (e.g. quantization, hard attention, non-differentiable norms).\n"
                    "Stopping. Consider gradient × input (multiply saliency by the raw\n"
                    "pixel values) or GradCAM on vision-encoder feature maps instead.",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(
                f"[smoke test] PASSED — meaningful spatial variation detected ({sal_range:.4f}).\n"
                f"             Proceeding to all {len(cases)} cases.",
                file=sys.stderr,
            )

    # 4. For each case × pipeline, extract and save
    for case_name, row in sorted(cases.items()):
        image_id = row["id"]
        img_path = img_root / row["img"]
        label_true = int(row["label_true"])

        if not img_path.exists():
            print(f"\n[error] Image not found: {img_path}", file=sys.stderr)
            continue

        print(f"\n[case {case_name}] id={image_id}  file={img_path.name}", file=sys.stderr)

        for pipeline in ("baseline", "categorized_v2"):
            out_path = out_root / pipeline / f"{image_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                prob, saliency = run_pipeline_with_saliency(
                    model, processor, img_path, pipeline,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                print(f"    [info] prob={prob:.4f}", file=sys.stderr)

                if saliency is None:
                    print("    [warn] No saliency computed — grid will be empty", file=sys.stderr)
                    grid: list = []
                else:
                    H, W = saliency.shape
                    grid = saliency_to_grid(saliency)
                    print(f"    [info] saliency grid: {H}×{W}", file=sys.stderr)

                result = {
                    "image_id": image_id,
                    "pipeline": pipeline,
                    "prob_pred": round(float(prob), 4),
                    "label_true": label_true,
                    "case": case_name,
                    "attention_grid": grid,
                }
                with open(out_path, "w") as fh:
                    json.dump(result, fh)
                print(f"    [saved] {out_path}", file=sys.stderr)

            except Exception as exc:
                print(f"    [error] {pipeline}: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    print("\n[done]", file=sys.stderr)


if __name__ == "__main__":
    main()

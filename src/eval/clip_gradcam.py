#!/usr/bin/env python3
"""
CLIP GradCAM heatmaps with per-image best-query selection.

For each of the top-10 True Positives (by prob_pred) from categorized_v2.csv:
  1. Score every candidate hate query against the image via CLIP cosine similarity.
  2. Select the query with the highest score — that is the hate type CLIP
     visually associates most strongly with this meme.
  3. Compute GradCAM on the patch-embedding Conv2d using only the best query.
  4. Save to report/heatmaps/clip/<id>.json with best_query and best_score fields.

Usage:
    python3 src/eval/clip_gradcam.py \\
        --img_dir data/eval_data \\
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
from transformers import CLIPProcessor, CLIPModel


HATE_QUERIES = [
    "hate speech",
    "hateful meme",
    "racist meme targeting a group",
    "meme mocking a religion",
    "antisemitic meme",
    "islamophobic content",
    "meme dehumanizing people",
    "offensive meme targeting minority",
    "meme using humor to spread hate",
    "historical genocide reference",
    "meme with racial slur",
    "sexist meme targeting women",
]

TOP_N = 10   # number of True Positives to process


def load_top_tp_cases(csv_path: Path, n: int = TOP_N) -> list[dict]:
    """
    Return the n True Positives (label_true=1, label_pred=1) with the
    highest prob_pred, sorted descending.
    """
    tps = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            try:
                if int(row["label_true"]) == 1 and int(row["label_pred"]) == 1:
                    tps.append(row)
            except (KeyError, ValueError):
                continue

    tps.sort(key=lambda r: float(r["prob_pred"]), reverse=True)
    return tps[:n]


def select_best_query(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_path: Path,
    queries: list[str],
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Score every query against the image and return:
      (best_query, best_score, ranked_list)

    ranked_list is [(query, cosine_score), ...] sorted highest-first.
    Cosine similarity is computed from the already-normalised embeddings
    that CLIPModel.forward() returns in image_embeds / text_embeds.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=queries, images=image, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Both embeds are L2-normalised inside CLIPModel.forward()
        scores = (outputs.image_embeds @ outputs.text_embeds.T).squeeze(0)  # [n_queries]

    scores_list = scores.cpu().float().tolist()
    ranked = sorted(zip(queries, scores_list), key=lambda x: x[1], reverse=True)
    best_query, best_score = ranked[0]
    return best_query, float(best_score), ranked


def compute_gradcam(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_path: Path,
    text_queries: list[str],
) -> np.ndarray:
    """
    Return a normalised GradCAM heatmap [16×16] as a float32 numpy array.

    We hook model.vision_model.embeddings.patch_embedding — the Conv2d that
    projects each 14×14 pixel patch to a 1024-dim feature vector.  Its output
    shape is [1, 1024, 16, 16], giving a clean 2D spatial map with one cell
    per image patch (for ViT-L/14 on 224×224 images).

    Why not the last encoder block?  In ViT-CLIP only the [CLS] token output
    from the final block feeds into the similarity score, so patch-position
    gradients at that block are identically zero.  The patch-embedding Conv2d
    receives non-zero gradients because the [CLS] token attends to all patch
    positions across all 24 encoder layers, propagating gradient back through
    the full stack to this layer.

    GradCAM: ReLU( sum_c( GAP_hw(grad_c) × feat_c ) )
    where c = channel, GAP_hw = global-average-pool over 16×16 spatial dims.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text_queries, images=image, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    _features: dict = {}
    _gradients: dict = {}

    def _fwd_hook(module, inp, out):
        _features["val"] = out                          # [1, 1024, 16, 16]

    def _bwd_hook(module, grad_in, grad_out):
        grad = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        _gradients["val"] = grad                        # [1, 1024, 16, 16]

    patch_embed = model.vision_model.embeddings.patch_embedding  # Conv2d
    fh = patch_embed.register_forward_hook(_fwd_hook)
    bh = patch_embed.register_full_backward_hook(_bwd_hook)

    try:
        model.zero_grad()
        with torch.enable_grad():
            outputs = model(**inputs)
            loss = outputs.logits_per_image.mean()      # scalar
            loss.backward()
    finally:
        fh.remove()
        bh.remove()

    if "val" not in _features or "val" not in _gradients:
        raise RuntimeError("Hooks did not fire — check model architecture.")

    features  = _features["val"]   # [1, 1024, 16, 16]
    gradients = _gradients["val"]  # [1, 1024, 16, 16]

    weights = gradients[0].mean(dim=(1, 2))                          # [1024]
    cam = (features[0] * weights[:, None, None]).sum(dim=0)         # [16, 16]
    cam = torch.relu(cam)

    cam_2d = cam.cpu().float().detach().numpy()
    lo, hi = float(cam_2d.min()), float(cam_2d.max())
    if hi > lo:
        cam_2d = (cam_2d - lo) / (hi - lo)
    else:
        cam_2d[:] = 0.0

    return cam_2d


def grid_to_json(arr: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in arr]


def print_ranking(ranked: list[tuple[str, float]], best_query: str) -> None:
    print("    Query ranking (cosine similarity):", file=sys.stderr)
    for i, (q, s) in enumerate(ranked, 1):
        marker = "  ← BEST" if q == best_query else ""
        print(f"      {i:2d}. {q:<45s}  {s:.4f}{marker}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="CLIP GradCAM heatmaps with per-image best-query selection"
    )
    parser.add_argument("--img_dir", required=True,
                        help="Dataset root; image paths resolved as <img_dir>/<img>")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--csv", default="report/categorized_v2.csv")
    parser.add_argument("--out_dir", default="report/heatmaps/clip")
    parser.add_argument("--model_name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--top_n", type=int, default=TOP_N,
                        help="Number of top True Positives to process")
    parser.add_argument("--skip_smoke_test", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    img_root = Path(args.img_dir)
    out_root = Path(args.out_dir)

    # 1. Load top-N True Positives
    print(f"[info] Loading top-{args.top_n} True Positives from {csv_path}", file=sys.stderr)
    cases = load_top_tp_cases(csv_path, n=args.top_n)
    if not cases:
        print("[error] No True Positive cases found in CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Selected {len(cases)} cases:", file=sys.stderr)
    for row in cases:
        print(
            f"  id={row['id']:5s}  prob={row['prob_pred']}  "
            f"text={repr(row['text'][:55])}",
            file=sys.stderr,
        )

    # 2. Load CLIP
    print(f"\n[info] Loading {args.model_name}", file=sys.stderr)
    processor = CLIPProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model     = CLIPModel.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device).eval()
    print(f"[info] Model loaded on {device}", file=sys.stderr)

    # 3. Smoke test on the first case
    if not args.skip_smoke_test:
        smoke_row  = cases[0]
        smoke_id   = smoke_row["id"]
        smoke_path = img_root / smoke_row["img"]
        print(f"\n[smoke test] id={smoke_id}  file={smoke_path.name}", file=sys.stderr)

        if not smoke_path.exists():
            print(f"[error] Smoke test image not found: {smoke_path}", file=sys.stderr)
            sys.exit(1)

        try:
            best_q, best_s, ranked = select_best_query(
                model, processor, smoke_path, HATE_QUERIES
            )
            print_ranking(ranked, best_q)
            cam = compute_gradcam(model, processor, smoke_path, [best_q])
        except Exception as exc:
            print(f"[error] Smoke test failed: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        sal_range = float(cam.max() - cam.min())
        H, W = cam.shape
        print(
            f"[smoke test] best_query={repr(best_q)}  score={best_s:.4f}\n"
            f"             grid {H}×{W}  normalised range={sal_range:.4f}",
            file=sys.stderr,
        )

        if sal_range < 0.1:
            print(
                f"\n[STOP] Smoke test: GradCAM map nearly flat "
                f"(max − min = {sal_range:.4f} < 0.1). Stopping.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"[smoke test] PASSED.\n"
            f"             Proceeding to all {len(cases)} cases.",
            file=sys.stderr,
        )

    # 4. Run all cases
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", file=sys.stderr)
    print("SUMMARY TABLE", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    for i, row in enumerate(cases, 1):
        image_id   = row["id"]
        img_path   = img_root / row["img"]
        label_true = int(row["label_true"])
        prob_pred  = float(row["prob_pred"])
        meme_text  = row["text"]

        if not img_path.exists():
            print(f"\n[error] Image not found: {img_path}", file=sys.stderr)
            continue

        print(
            f"\n[{i:2d}/{len(cases)}] id={image_id}  prob_pred={prob_pred:.2f}  "
            f"text={repr(meme_text[:55])}",
            file=sys.stderr,
        )
        out_path = out_root / f"{image_id}.json"

        try:
            # Select best query
            best_q, best_s, ranked = select_best_query(
                model, processor, img_path, HATE_QUERIES
            )
            print_ranking(ranked, best_q)

            # GradCAM for best query only
            cam = compute_gradcam(model, processor, img_path, [best_q])
            H, W = cam.shape
            sal_range = float(cam.max() - cam.min())
            print(
                f"    GradCAM: grid {H}×{W}  range={sal_range:.4f}  "
                f"best_query={repr(best_q)}  score={best_s:.4f}",
                file=sys.stderr,
            )

            result = {
                "image_id":       image_id,
                "pipeline":       "clip_gradcam",
                "prob_pred":      round(prob_pred, 4),
                "label_true":     label_true,
                "case":           "TP",
                "best_query":     best_q,
                "best_score":     round(best_s, 4),
                "query_ranking":  [[q, round(s, 4)] for q, s in ranked],
                "attention_grid": grid_to_json(cam),
            }
            with open(out_path, "w") as fh:
                json.dump(result, fh)
            print(f"    [saved] {out_path}", file=sys.stderr)

        except Exception as exc:
            print(f"    [error] {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\n{'='*70}", file=sys.stderr)
    print("[done]", file=sys.stderr)


if __name__ == "__main__":
    main()

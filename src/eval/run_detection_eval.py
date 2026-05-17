"""
Detection evaluation for UnHateMemeDL.

Two phases:
  Phase 1 (slow, GPU) : run_inference  — queries the VLM for every image,
                                          appends rows to a CSV (resumable).
  Phase 2 (fast, CPU) : compute_metrics — loads the CSV, prints AUROC / F1 / Accuracy.

Usage
-----
  # Full run (inference + metrics):
  python src/eval/run_detection_eval.py \
      --jsonl  data/eval_data/eval_490_balanced.jsonl \
      --img_dir <root-of-dataset> \
      --output  report/detection_predictions.csv

  # Metrics only (from an existing CSV):
  python src/eval/run_detection_eval.py \
      --jsonl  data/eval_data/eval_490_balanced.jsonl \
      --output  report/detection_predictions.csv \
      --metrics_only

  # Add per-modality F1 (runs detect_hate_type on every ground-truth hateful image):
  python src/eval/run_detection_eval.py ... --modality_analysis

Notes
-----
- --img_dir should be the dataset root; image paths from the JSONL
  (e.g. "img/57823.png") are resolved relative to it.
- Progress is flushed after every image so a crash is safely resumable.
"""

import sys
import json
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Make pipeline modules importable
sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

from vlm import (
    instantiate_vlm,
    load_cls_head,
    detect_hateful_meme,
    detect_hateful_meme_cls_head,
    detect_hate_type,
)
from utils import parse_hateful_response, parse_hate_type_response

FIELDNAMES = [
    "id", "img", "label_true", "text",
    "prob_pred", "label_pred", "classification", "description",
    "modality_type", "error",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {int(row["id"]) for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Phase 1 — inference
# ---------------------------------------------------------------------------

def run_inference(args):
    samples  = load_jsonl(Path(args.jsonl))
    img_root = Path(args.img_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_existing_ids(out_path)
    todo = [s for s in samples if int(s["id"]) not in done]

    if not todo:
        print("[info] All images already processed. Skipping inference.", file=sys.stderr)
        return

    print(f"[info] {len(done)} already done, {len(todo)} remaining.", file=sys.stderr)

    vlm, processor = instantiate_vlm(args.vlm_name, args.cache_dir, args.adapter_path)
    print(f"[info] VLM loaded on {vlm.device}", file=sys.stderr)

    # Load classification head if provided — detection becomes a forward pass,
    # no generation. The generative path is used as fallback when head is absent.
    cls_head = None
    if args.cls_head_path:
        cls_head = load_cls_head(vlm, args.cls_head_path)
        print(f"[info] Detection mode: classification head ({args.cls_head_path})",
              file=sys.stderr)
    else:
        print("[info] Detection mode: generative (VLM output parsing)", file=sys.stderr)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, sample in enumerate(todo):
            img_path = img_root / sample["img"]
            print(f"\n[{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

            row = {
                "id":             sample["id"],
                "img":            sample["img"],
                "label_true":     sample["label"],
                "text":           sample.get("text", "").replace("\n", " "),
                "prob_pred":      "",
                "label_pred":     "",
                "classification": "",
                "description":    "",
                "modality_type":  "",
                "error":          "",
            }

            # --- detection --------------------------------------------------
            try:
                if cls_head is not None:
                    # Forward pass only — no generation, no JSON to parse.
                    # description stays empty (head produces no explanation).
                    is_hateful, prob = detect_hateful_meme_cls_head(
                        vlm, processor, cls_head, img_path
                    )
                else:
                    # Generative path — VLM produces a JSON blob to parse.
                    raw = detect_hateful_meme(vlm, processor, img_path)
                    is_hateful, prob, description = parse_hateful_response(raw)
                    row["description"] = description.replace("\n", " ")

                row["prob_pred"]      = prob
                row["label_pred"]     = 1 if prob >= 0.5 else 0
                row["classification"] = "hateful" if is_hateful else "non-hateful"
                print(f"         prob={prob:.3f}  → {'HATEFUL' if is_hateful else 'ok'}",
                      file=sys.stderr)
            except Exception as e:
                print(f"[error] Detection failed: {e}", file=sys.stderr)
                row["error"] = str(e)
                writer.writerow(row)
                f.flush()
                continue

            # --- per-modality (only on ground-truth hateful) -----------------
            if args.modality_analysis and int(sample["label"]) == 1:
                try:
                    raw_type = detect_hate_type(vlm, processor, img_path)
                    row["modality_type"] = parse_hate_type_response(raw_type)
                except Exception as e:
                    print(f"[warn] Modality detection failed: {e}", file=sys.stderr)
                    row["modality_type"] = "parse_error"

            writer.writerow(row)
            f.flush()

    print(f"\n[info] Inference complete → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 2 — metrics
# ---------------------------------------------------------------------------

def compute_metrics(args):
    out_path = Path(args.output)
    if not out_path.exists():
        print(f"[error] No results file at {out_path}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    valid   = [r for r in rows if r["prob_pred"] and not r["error"]]
    skipped = len(rows) - len(valid)
    if skipped:
        print(f"[warn] {skipped} rows skipped (inference error).\n")

    y_true = np.array([int(r["label_true"]) for r in valid])
    y_prob = np.array([float(r["prob_pred"]) for r in valid])
    y_pred = np.array([int(r["label_pred"]) for r in valid])

    auroc   = roc_auc_score(y_true, y_prob)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc     = accuracy_score(y_true, y_pred)

    sep = "=" * 52
    print(sep)
    print("DETECTION METRICS")
    print(sep)
    # Show which detection mode produced this CSV so comparisons are unambiguous
    has_descriptions = any(r.get("description") for r in valid)
    mode = "generative (VLM)" if has_descriptions else "classification head"
    print(f"  Mode       : {mode}")
    print(f"  N          : {len(valid)}  ({y_true.sum()} hateful / {(~y_true.astype(bool)).sum()} non-hateful)")
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  Macro-F1   : {macro_f1:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["non-hateful", "hateful"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':20} pred=0   pred=1")
    print(f"  {'true=0 (non-hateful)':20}  {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  {'true=1 (hateful)':20}  {cm[1,0]:5d}    {cm[1,1]:5d}")

    # --- per-modality F1 ---------------------------------------------------
    mod_rows = [r for r in valid
                if r.get("modality_type") and r["modality_type"] not in ("", "parse_error")]
    if mod_rows:
        print("\n  Per-modality F1 (ground-truth hateful images only):")
        for mod in ("unimodal-hate", "multimodal-hate"):
            subset = [r for r in mod_rows if mod in r["modality_type"]]
            if not subset:
                continue
            yt = np.array([int(r["label_true"]) for r in subset])
            yp = np.array([int(r["label_pred"]) for r in subset])
            print(f"    {mod:20}: F1={f1_score(yt, yp, zero_division=0):.4f}  (n={len(subset)})")

    print(sep)

    # save summary next to the CSV
    summary_path = out_path.with_suffix(".metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"AUROC={auroc:.4f}  Macro-F1={macro_f1:.4f}  Accuracy={acc:.4f}\n")
    print(f"\n  Summary saved → {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detection evaluation for UnHateMemeDL")
    parser.add_argument("--jsonl",     required=True,  help="Path to eval JSONL file")
    parser.add_argument("--img_dir",   default=None,   help="Dataset root (images resolved as <img_dir>/<img field>)")
    parser.add_argument("--output",    default="report/detection_predictions.csv")
    parser.add_argument("--vlm_name",  default="google/gemma-4-31B-it")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--adapter_path", default=None,
                        help="LoRA adapter directory (checkpoints/detect/adapter_detect). "
                             "Mutually exclusive with --cls_head_path.")
    parser.add_argument("--cls_head_path", default=None,
                        help="Path to classifier.pt from train_cls_head.py "
                             "(e.g. checkpoints/cls_head/best_classifier.pt). "
                             "When set, detection uses a forward pass instead of generation.")
    parser.add_argument("--metrics_only",      action="store_true", help="Skip inference, only compute metrics from existing CSV")
    parser.add_argument("--modality_analysis", action="store_true", help="Run detect_hate_type on hateful images for per-modality F1")
    args = parser.parse_args()

    if args.adapter_path and args.cls_head_path:
        parser.error("--adapter_path and --cls_head_path are mutually exclusive")

    if not args.metrics_only:
        if not args.img_dir:
            parser.error("--img_dir is required unless --metrics_only is set")
        run_inference(args)

    compute_metrics(args)


if __name__ == "__main__":
    main()

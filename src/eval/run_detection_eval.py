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

from vlm import instantiate_vlm, detect_hateful_meme, detect_hate_type
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

    vlm, processor = instantiate_vlm(args.vlm_name, args.cache_dir)
    print(f"[info] VLM loaded on {vlm.device}", file=sys.stderr)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, sample in enumerate(todo):
            img_path = img_root / sample["img"]
            print(f"\n[{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

            row = {
                "id":            sample["id"],
                "img":           sample["img"],
                "label_true":    sample["label"],
                "text":          sample.get("text", "").replace("\n", " "),
                "prob_pred":     "",
                "label_pred":    "",
                "classification": "",
                "description":   "",
                "modality_type": "",
                "error":         "",
            }

            # --- detection --------------------------------------------------
            try:
                raw = detect_hateful_meme(vlm, processor, img_path)
                is_hateful, prob, description = parse_hateful_response(raw)
                row["prob_pred"]     = prob
                row["label_pred"]    = 1 if prob >= 0.5 else 0
                row["classification"] = "hateful" if is_hateful else "non-hateful"
                row["description"]   = description.replace("\n", " ")
                print(f"         prob={prob:.3f}  → {'HATEFUL' if is_hateful else 'ok'}", file=sys.stderr)
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
    parser.add_argument("--vlm_name",  default="Qwen/Qwen3.6-27B")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--metrics_only",      action="store_true", help="Skip inference, only compute metrics from existing CSV")
    parser.add_argument("--modality_analysis", action="store_true", help="Run detect_hate_type on hateful images for per-modality F1")
    args = parser.parse_args()

    if not args.metrics_only:
        if not args.img_dir:
            parser.error("--img_dir is required unless --metrics_only is set")
        run_inference(args)

    compute_metrics(args)


if __name__ == "__main__":
    main()

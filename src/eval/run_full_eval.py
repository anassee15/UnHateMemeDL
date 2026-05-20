#!/usr/bin/env python3
"""
run_full_eval.py — Full pipeline evaluation across multiple VLMs.

Iterates over every model in MODELS and runs three phases:
  Phase 1  Detection   — VLM inference on all eval images, timed per image.
  Phase 2  Mitigation  — full pipeline on hateful images (VLM prompt + diffusion), timed per image.
  Phase 3  Metrics     — detection metrics (AUROC, F1, confusion matrix …)
                         + mitigation quality metrics (BERTScore, CLIPScore, SSIM, MPS).

Outputs
-------
  <output_root>/<model_slug>/detection_predictions.csv   per-image predictions + t_detect_s
  <output_root>/<model_slug>/mitigation_results.csv      per-image scores + t_prompt_s, t_diffusion_s
  <output_root>/<model_slug>/mitigated/                  mitigated PNGs + intermediate JSONs
  <output_root>/summary.csv                              one row per model, all aggregated metrics

Both phases are resumable: already-processed images are skipped.
If a model's summary row already exists in summary.csv, the whole model is skipped.

Usage
-----
  # Full run (all models, all phases):
  python src/eval/run_full_eval.py

  # Detection only (skip mitigation):
  python src/eval/run_full_eval.py --skip_mitigation

  # Recompute metrics from existing CSVs (no GPU needed):
  python src/eval/run_full_eval.py --metrics_only
"""

import sys
import json
import csv
import time
import argparse
import re
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

from vlm import instantiate_vlm, detect_hateful_meme, detect_hateful_meme_cls_head, get_diffusion_prompt, load_cls_head
from diffusion import instantiate_diffusion, mitigate_image
from utils import parse_hateful_response, parse_prompt_generation

# ============================================================
# CONFIGURATION — edit before running
# ============================================================

# ============================================================
# EXPERIMENT CONFIGURATION — N repetitions × eval types × models
# ============================================================

N_REPS = 3   # repetitions per (eval_type, model) for statistical significance

_GEMMA4 = "google/gemma-4-31B-it"
_QWEN36 = "Qwen/Qwen3.6-27B"
_QWEN25 = "Qwen/Qwen2.5-VL-7B-Instruct"

EXPERIMENTS = {
    "baseline": {
        "run_mitigation": True,
        "models": [
            {"vlm_name": _GEMMA4, "adapter_path": None, "mitigation_adapter_path": None, "cls_head_path": None},
            {"vlm_name": _QWEN36, "adapter_path": None, "mitigation_adapter_path": None, "cls_head_path": None},
            {"vlm_name": _QWEN25, "adapter_path": None, "mitigation_adapter_path": None, "cls_head_path": None},
        ],
    },
    "cls_head": {
        "run_mitigation": False,
        "models": [
            {"vlm_name": _GEMMA4, "adapter_path": None, "mitigation_adapter_path": None, "cls_head_path": "checkpoints/cls_head/gemma4/best_classifier.pt"},
            {"vlm_name": _QWEN25, "adapter_path": None, "mitigation_adapter_path": None, "cls_head_path": "checkpoints/cls_head/qwen25/best_classifier.pt"},
        ],
    },
    "lora_detect": {
        "run_mitigation": False,
        "models": [
            {"vlm_name": _GEMMA4, "adapter_path": "checkpoints/detect/gemma4", "mitigation_adapter_path": None, "cls_head_path": None},
            {"vlm_name": _QWEN25, "adapter_path": "checkpoints/detect/qwen25", "mitigation_adapter_path": None, "cls_head_path": None},
        ],
    },
    "lora_mitigate": {
        "run_mitigation": True,
        "models": [
            {"vlm_name": _GEMMA4, "adapter_path": "checkpoints/detect/gemma4", "mitigation_adapter_path": "checkpoints/mitigate/gemma4", "cls_head_path": None},
            {"vlm_name": _QWEN25, "adapter_path": "checkpoints/detect/qwen25", "mitigation_adapter_path": "checkpoints/mitigate/qwen25", "cls_head_path": None},
        ],
    },
}

JSONL_PATH   = "data/eval_data/eval_490_balanced.jsonl"
IMG_DIR      = "data/hateful-meme"   # Images root (JSONLs reference img/NNNN.png relative to this)
OUTPUT_ROOT  = "report/full_eval"
CACHE_DIR    = None                   # set to a local path if you use a HF cache dir
DIFFUSION_MODEL = "black-forest-labs/FLUX.2-klein-9B"

# ============================================================
# CSV SCHEMAS
# ============================================================

DET_FIELDNAMES = [
    "id", "img", "label_true", "text",
    "prob_pred", "label_pred", "classification", "description",
    "modality_type", "error", "t_detect_s",
]

MIT_FIELDNAMES = [
    "id", "img", "label_true",
    "prob_before", "prob_after",
    "detoxify_before", "detoxify_after",
    "hate_source", "hate_location",
    "original_text", "replacement_text", "diffusion_prompt",
    "bertscore_f1", "clip_score", "ssim", "mps",
    "mitigated_path", "error",
    "t_prompt_s", "t_diffusion_s",
]

SUMMARY_FIELDNAMES = [
    # identity
    "eval_type", "rep", "vlm_name", "model_slug",
    # detection — dataset stats
    "n_total", "n_hateful", "n_nonhateful", "n_errors",
    # detection — quality
    "auroc", "macro_f1", "accuracy",
    "precision_hateful", "recall_hateful", "f1_hateful",
    "tp", "tn", "fp", "fn",
    # detection — timing (seconds / image)
    "det_mean_s", "det_std_s", "det_median_s", "det_total_s",
    # mitigation — dataset stats
    "n_mitigated",
    # mitigation — toxicity reduction
    "mean_prob_before", "mean_prob_after", "mean_tr_pct", "pct_nonhateful_after",
    # mitigation — content preservation
    "mean_bertscore_f1", "mean_clip_score", "mean_ssim", "mean_mps",
    # mitigation — timing (seconds / image)
    "prompt_mean_s", "prompt_std_s",
    "diffusion_mean_s", "diffusion_std_s",
    "mit_total_mean_s",
]


# ============================================================
# UTILITIES
# ============================================================

def slugify(model_name: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_ids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {int(row["id"]) for row in csv.DictReader(f)}


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _try_import(pkg: str, pip_name: str | None = None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[warn] '{pip_name or pkg}' not installed — skipping. "
              f"pip install {pip_name or pkg}", file=sys.stderr)
        return None


# ============================================================
# STEP 1 — Detection inference
# Run the VLM on every image in the eval set.
# Times each call and appends a row to <model_dir>/detection_predictions.csv.
# Resumable: images already in the CSV are skipped.
# ============================================================

def run_detection(vlm, processor, samples: list[dict], img_root: Path, out_path: Path, *, cls_head=None) -> None:
    done = load_existing_ids(out_path)
    todo = [s for s in samples if int(s["id"]) not in done]

    if not todo:
        print("[step1] All images already processed — skipping detection.", file=sys.stderr)
        return

    print(f"[step1] Detection: {len(done)} done, {len(todo)} remaining.", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(todo):
        img_path = img_root / sample["img"]
        print(f"  [{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr, end="  ")

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
            "t_detect_s":     "",
        }

        t0 = time.perf_counter()
        try:
            if cls_head is not None:
                # Classification head path — pure forward pass
                is_hateful, prob = detect_hateful_meme_cls_head(vlm, processor, cls_head, img_path)
                row["t_detect_s"] = f"{time.perf_counter() - t0:.3f}"
                row["prob_pred"]      = prob
                row["label_pred"]     = 1 if prob >= 0.5 else 0
                row["classification"] = "hateful" if is_hateful else "non-hateful"
                row["description"]    = ""
                print(f"prob={prob:.3f} t={row['t_detect_s']}s (cls_head)", file=sys.stderr)
            else:
                # Generative VLM path (baseline or LoRA)
                raw = detect_hateful_meme(vlm, processor, img_path)
                row["t_detect_s"] = f"{time.perf_counter() - t0:.3f}"
                is_hateful, prob, description = parse_hateful_response(raw)
                row["prob_pred"]      = prob
                row["label_pred"]     = 1 if prob >= 0.5 else 0
                row["classification"] = "hateful" if is_hateful else "non-hateful"
                row["description"]    = description.replace("\n", " ")
                print(f"prob={prob:.3f} t={row['t_detect_s']}s", file=sys.stderr)
        except Exception as e:
            row["t_detect_s"] = f"{time.perf_counter() - t0:.3f}"
            row["error"] = str(e)
            print(f"ERROR: {e}", file=sys.stderr)

        append_csv_row(out_path, DET_FIELDNAMES, row)

    print(f"[step1] Detection complete → {out_path}", file=sys.stderr)


# ============================================================
# STEP 2 — Detection metrics
# Reads the predictions CSV and computes:
#   AUROC, Macro-F1, Accuracy, per-class precision/recall/F1,
#   confusion matrix (TP, TN, FP, FN), timing statistics.
# Returns a dict of scalars that will be merged into the summary row.
# ============================================================

def compute_detection_metrics(det_path: Path) -> dict:
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        classification_report, confusion_matrix,
        precision_score, recall_score,
    )

    with open(det_path, newline="") as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r.get("prob_pred") and not r.get("error")]
    n_errors = len(rows) - len(valid)
    print(f"[step2] {len(valid)} valid rows, {n_errors} errors.", file=sys.stderr)

    y_true = np.array([int(r["label_true"]) for r in valid])
    y_prob = np.array([float(r["prob_pred"]) for r in valid])
    y_pred = np.array([int(r["label_pred"]) for r in valid])

    auroc    = roc_auc_score(y_true, y_prob)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc      = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_hat   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Timing
    times = [float(r["t_detect_s"]) for r in valid if r.get("t_detect_s")]
    t_arr = np.array(times) if times else np.array([float("nan")])

    sep = "=" * 52
    print(f"\n{sep}\nDETECTION METRICS\n{sep}", file=sys.stderr)
    print(f"  N={len(valid)}  hateful={y_true.sum()}  non-hateful={(~y_true.astype(bool)).sum()}")
    print(f"  AUROC    : {auroc:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Det time : mean={t_arr.mean():.2f}s  std={t_arr.std():.2f}s")
    print(sep)

    return {
        "n_total":      len(valid),
        "n_hateful":    int(y_true.sum()),
        "n_nonhateful": int((~y_true.astype(bool)).sum()),
        "n_errors":     n_errors,
        "auroc":        round(auroc, 4),
        "macro_f1":     round(macro_f1, 4),
        "accuracy":     round(acc, 4),
        "precision_hateful": round(prec, 4),
        "recall_hateful":    round(rec, 4),
        "f1_hateful":        round(f1_hat, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "det_mean_s":   round(float(t_arr.mean()), 3),
        "det_std_s":    round(float(t_arr.std()), 3),
        "det_median_s": round(float(np.median(t_arr)), 3),
        "det_total_s":  round(float(t_arr.sum()), 1),
    }


# ============================================================
# STEP 3 — Mitigation
# For each ground-truth hateful image:
#   a) Ask the VLM to produce a mitigation plan (prompt + text edits) — timed.
#   b) Apply the diffusion model to generate the mitigated image — timed.
# Saves the mitigated PNG and the intermediate JSON to <model_dir>/mitigated/.
# Appends a row to mitigation_results.csv with t_prompt_s and t_diffusion_s.
# Resumable: images whose PNG already exists are skipped.
# ============================================================

def run_mitigation(
    vlm, processor, diffusion_model,
    samples: list[dict], img_root: Path, det_path: Path,
    out_dir: Path, mit_csv_path: Path,
) -> None:
    # Load detection results to reuse prob_before (avoids a redundant VLM call)
    det: dict = {}
    if det_path.exists():
        with open(det_path, newline="") as f:
            det = {int(r["id"]): r for r in csv.DictReader(f)}

    hateful = [s for s in samples if int(s["label"]) == 1]
    done_ids = load_existing_ids(mit_csv_path)
    todo = [s for s in hateful if int(s["id"]) not in done_ids]

    if not todo:
        print("[step3] All hateful images already mitigated — skipping.", file=sys.stderr)
        return

    print(f"[step3] Mitigation: {len(hateful) - len(todo)} done, {len(todo)} remaining.", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    for i, sample in enumerate(todo):
        sid      = sample["id"]
        img_path = img_root / sample["img"]
        mit_path = out_dir / f"{sid}_mitigated.png"
        jpath    = out_dir / f"{sid}_intermediate.json"
        print(f"  [{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

        row = {k: "" for k in MIT_FIELDNAMES}
        row["id"]         = sid
        row["img"]        = sample["img"]
        row["label_true"] = sample["label"]

        # Reuse prob_before from detection CSV when available
        det_row = det.get(int(sid))
        if det_row and det_row.get("prob_pred"):
            row["prob_before"] = det_row["prob_pred"]
        row["mitigated_path"] = str(mit_path)

        try:
            image = Image.open(img_path).convert("RGB")
            prob_before = float(row["prob_before"]) if row["prob_before"] else 1.0

            if prob_before < 0.2:
                # Too low confidence — save original unchanged, no diffusion needed
                image.save(mit_path)
                jpath.write_text(json.dumps({"skipped": "prob < 0.2", "prob": prob_before}))
                row["t_prompt_s"] = "0"
                row["t_diffusion_s"] = "0"
                print(f"    prob={prob_before:.3f} < 0.2 → skipped", file=sys.stderr)
                append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)
                continue

            # 3a — VLM: generate mitigation plan
            t_p = time.perf_counter()
            raw_prompt = get_diffusion_prompt(vlm, processor, img_path)
            mitigation = parse_prompt_generation(raw_prompt)
            row["t_prompt_s"] = f"{time.perf_counter() - t_p:.3f}"
            jpath.write_text(json.dumps(mitigation, ensure_ascii=False, indent=2))

            row["hate_source"]      = mitigation.get("hate_source", "")
            row["hate_location"]    = mitigation.get("hate_location", "")
            row["original_text"]    = (mitigation.get("original_text") or "").replace("\n", "\\n")
            row["replacement_text"] = (mitigation.get("replacement_text") or "").replace("\n", "\\n")
            row["diffusion_prompt"] = mitigation.get("diffusion_prompt", "")

            # 3b — Diffusion: apply mitigation to the image
            t_d = time.perf_counter()
            mitigated = mitigate_image(diffusion_model, image, mitigation, generator=generator)
            row["t_diffusion_s"] = f"{time.perf_counter() - t_d:.3f}"
            mitigated.save(mit_path)

            print(f"    prompt={row['t_prompt_s']}s  diffusion={row['t_diffusion_s']}s", file=sys.stderr)

        except Exception as e:
            row["error"] = str(e)
            print(f"    ERROR: {e}", file=sys.stderr)
            jpath.write_text(json.dumps({"error": str(e)}))

        append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)

    print(f"[step3] Mitigation complete → {out_dir}", file=sys.stderr)


# ============================================================
# STEP 3a — Mitigation: VLM prompt generation only (two-phase optimization)
# Generates mitigation plans for all hateful images, saves to _intermediate.json.
# Does NOT apply diffusion — that happens in a separate phase to minimize peak GPU memory.
# (VLM + diffusion never loaded simultaneously.)
# ============================================================

def run_mitigation_prompts(
    vlm, processor,
    samples: list[dict], img_root: Path, det_path: Path,
    out_dir: Path,
) -> None:
    """Generate VLM mitigation prompts only, saving to _intermediate.json files."""
    # Load detection results to reuse prob_before
    det: dict = {}
    if det_path is not None and det_path.exists():
        with open(det_path, newline="") as f:
            det = {int(r["id"]): r for r in csv.DictReader(f)}

    hateful = [s for s in samples if int(s["label"]) == 1]

    # Check which JSONs are already generated (skip if exist)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in hateful if not (out_dir / f"{s['id']}_intermediate.json").exists()]

    if not todo:
        print(f"[step3a] All mitigation prompts already generated — skipping.", file=sys.stderr)
        return

    print(f"[step3a] Generating prompts: {len(hateful) - len(todo)} done, {len(todo)} remaining.", file=sys.stderr)

    for i, sample in enumerate(todo):
        sid = sample["id"]
        img_path = img_root / sample["img"]
        jpath = out_dir / f"{sid}_intermediate.json"

        try:
            prob_before = 1.0
            det_row = det.get(int(sid))
            if det_row and det_row.get("prob_pred"):
                prob_before = float(det_row["prob_pred"])

            if prob_before < 0.2:
                # Too low confidence — skip diffusion
                jpath.write_text(json.dumps({"skipped": "prob < 0.2", "prob": prob_before}))
                print(f"  [{i+1}/{len(todo)}] {img_path.name}: prob={prob_before:.3f} < 0.2 → skip", file=sys.stderr)
                continue

            # Generate mitigation plan
            raw_prompt = get_diffusion_prompt(vlm, processor, img_path)
            mitigation = parse_prompt_generation(raw_prompt)
            jpath.write_text(json.dumps(mitigation, ensure_ascii=False, indent=2))
            print(f"  [{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

        except Exception as e:
            jpath.write_text(json.dumps({"error": str(e)}))
            print(f"  [{i+1}/{len(todo)}] {img_path.name}: ERROR: {e}", file=sys.stderr)

    print(f"[step3a] Prompt generation complete → {out_dir}", file=sys.stderr)


# ============================================================
# STEP 3b — Mitigation: Diffusion inference only (two-phase optimization)
# Reads mitigation plans from _intermediate.json files and applies diffusion.
# VLM is not needed; diffusion model runs alone to minimize peak GPU memory.
# ============================================================

def run_mitigation_diffusion(
    diffusion_model,
    samples: list[dict], img_root: Path,
    mit_csv_path: Path, out_dir: Path,
    generator: torch.Generator,
    det_path: Path | None = None,
) -> None:
    """Apply diffusion model to hateful images, reading mitigation plans from _intermediate.json."""
    det: dict = {}
    if det_path is not None and det_path.exists():
        with open(det_path, newline="") as f:
            det = {int(r["id"]): r for r in csv.DictReader(f)}

    hateful = [s for s in samples if int(s["label"]) == 1]

    # Load CSV to check which images are already fully processed (have prob_after or error)
    done_ids = load_existing_ids(mit_csv_path)
    todo = [s for s in hateful if int(s["id"]) not in done_ids]

    if not todo:
        print(f"[step3b] All images already diffused — skipping.", file=sys.stderr)
        return

    print(f"[step3b] Diffusion: {len(hateful) - len(todo)} done, {len(todo)} remaining.", file=sys.stderr)

    for i, sample in enumerate(todo):
        sid = sample["id"]
        img_path = img_root / sample["img"]
        mit_path = out_dir / f"{sid}_mitigated.png"
        jpath = out_dir / f"{sid}_intermediate.json"

        print(f"  [{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

        row = {k: "" for k in MIT_FIELDNAMES}
        row["id"]         = sid
        row["img"]        = sample["img"]
        row["label_true"] = sample["label"]
        row["mitigated_path"] = str(mit_path)
        det_row = det.get(int(sid))
        if det_row and det_row.get("prob_pred"):
            row["prob_before"] = det_row["prob_pred"]

        try:
            # Load intermediate JSON
            if not jpath.exists():
                row["error"] = "intermediate.json not found"
                append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)
                print(f"    ERROR: intermediate.json not found", file=sys.stderr)
                continue

            mitigation_data = json.loads(jpath.read_text())

            # Check if this was skipped or errored during prompt generation
            if mitigation_data.get("skipped"):
                row["t_prompt_s"] = "0"
                row["t_diffusion_s"] = "0"
                image = Image.open(img_path).convert("RGB")
                image.save(mit_path)
                print(f"    Skipped (prob < 0.2)", file=sys.stderr)
                append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)
                continue

            if "error" in mitigation_data:
                row["error"] = mitigation_data["error"]
                append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)
                print(f"    Error from prompt gen: {mitigation_data['error']}", file=sys.stderr)
                continue

            # Apply diffusion
            image = Image.open(img_path).convert("RGB")
            t_d = time.perf_counter()
            mitigated = mitigate_image(diffusion_model, image, mitigation_data, generator=generator)
            row["t_diffusion_s"] = f"{time.perf_counter() - t_d:.3f}"
            mitigated.save(mit_path)

            # Populate mitigation fields from JSON
            row["hate_source"]      = mitigation_data.get("hate_source", "")
            row["hate_location"]    = mitigation_data.get("hate_location", "")
            row["original_text"]    = (mitigation_data.get("original_text") or "").replace("\n", "\\n")
            row["replacement_text"] = (mitigation_data.get("replacement_text") or "").replace("\n", "\\n")
            row["diffusion_prompt"] = mitigation_data.get("diffusion_prompt", "")
            row["t_prompt_s"]       = "0"  # prompt gen was done in earlier phase

            print(f"    diffusion={row['t_diffusion_s']}s", file=sys.stderr)

        except Exception as e:
            row["error"] = str(e)
            print(f"    ERROR: {e}", file=sys.stderr)

        append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)

    print(f"[step3b] Diffusion complete → {out_dir}", file=sys.stderr)


# ============================================================
# STEP 4 — VLM judge on mitigated images
# Re-runs the VLM on every mitigated PNG to get prob_after.
# This measures how much the mitigation actually reduced hatefulness
# according to the model itself.
# Updates prob_after in the mitigation results CSV.
# ============================================================

def run_judge(vlm, processor, mit_csv_path: Path, out_dir: Path) -> None:
    if not mit_csv_path.exists():
        print("[step4] No mitigation CSV found — skipping judge.", file=sys.stderr)
        return

    with open(mit_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Find rows that are missing prob_after
    to_judge = [r for r in rows if not r.get("prob_after") and not r.get("error")]
    if not to_judge:
        print("[step4] All rows already judged — skipping.", file=sys.stderr)
        return

    print(f"[step4] Judging {len(to_judge)} mitigated images.", file=sys.stderr)

    id_to_row = {r["id"]: r for r in rows}
    for i, row in enumerate(to_judge):
        mit_path = Path(row.get("mitigated_path", out_dir / f"{row['id']}_mitigated.png"))
        print(f"  [{i+1}/{len(to_judge)}] {mit_path.name}", file=sys.stderr, end="  ")
        if not mit_path.exists():
            print("not found — skip", file=sys.stderr)
            continue
        try:
            raw = detect_hateful_meme(vlm, processor, mit_path)
            _, prob_after, _ = parse_hateful_response(raw)
            id_to_row[row["id"]]["prob_after"] = f"{prob_after:.4f}"
            print(f"before={row['prob_before']}  after={prob_after:.3f}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            id_to_row[row["id"]]["error"] = str(e)

    # Rewrite the CSV with updated prob_after values
    mit_csv_path.write_text("")  # truncate
    with open(mit_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MIT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            # Fill in any new fields from extended schema
            writer.writerow({k: row.get(k, "") for k in MIT_FIELDNAMES})

    print(f"[step4] Judge complete → {mit_csv_path}", file=sys.stderr)


# ============================================================
# STEP 5 — Mitigation metrics
# Reads the mitigation results CSV and computes:
#   Axis A — Toxicity reduction (TR%, % images now non-hateful)
#   Axis B — Content preservation (BERTScore, CLIPScore, SSIM, MPS)
#   Timing statistics for prompt generation and diffusion.
# Returns a dict that will be merged into the summary row.
# ============================================================

def compute_mitigation_metrics(mit_csv_path: Path, img_root: Path) -> dict:
    if not mit_csv_path.exists():
        return {}

    with open(mit_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    hateful_rows = [r for r in rows
                    if r.get("prob_after")
                    and not r.get("error")
                    and int(r.get("label_true", 0)) == 1]

    if not hateful_rows:
        print("[step5] No valid hateful rows in mitigation CSV.", file=sys.stderr)
        return {}

    # --- Axis A: Toxicity Reduction ---
    prob_after = np.array([float(r["prob_after"]) for r in hateful_rows], dtype=float)
    pct_nonhateful = (prob_after < 0.5).mean() * 100

    # TR% only when prob_before is available (not in mitigation_only runs)
    tr_rows = [r for r in hateful_rows if r.get("prob_before")]
    tr_per_img = np.array([])
    if tr_rows:
        prob_before = np.array([float(r["prob_before"]) for r in tr_rows], dtype=float)
        prob_after_tr = np.array([float(r["prob_after"]) for r in tr_rows], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            tr_per_img = np.where(prob_before > 0, (prob_before - prob_after_tr) / prob_before, 0.0)

    # --- Axis B: Content Preservation ---
    text_rows = [r for r in hateful_rows if r.get("original_text") and r.get("replacement_text")]
    orig_texts = [r["original_text"].replace("\\n", " ") for r in text_rows]
    repl_texts = [r["replacement_text"].replace("\\n", " ") for r in text_rows]

    # BERTScore (original meme text vs. replacement text)
    mean_bert = float("nan")
    bs = _try_import("bert_score")
    if bs and text_rows:
        _, _, F = bs.score(repl_texts, orig_texts, lang="en", verbose=False)
        mean_bert = float(F.mean())

    # CLIPScore (mitigated image <-> replacement text)
    mean_clip = float("nan")
    clip_rows = [r for r in hateful_rows if r.get("replacement_text") and r.get("mitigated_path")]
    tf = _try_import("transformers")
    if tf and clip_rows:
        clip_model = tf.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_proc  = tf.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        clip_scores = []
        for r in clip_rows:
            try:
                img  = Image.open(r["mitigated_path"]).convert("RGB")
                txt  = r["replacement_text"].replace("\\n", " ")
                inp  = clip_proc(text=[txt], images=[img], return_tensors="pt", padding=True)
                with torch.no_grad():
                    out = clip_model(**inp)
                ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
                clip_scores.append((ie * te).sum().item())
            except Exception:
                clip_scores.append(float("nan"))
        valid_clips = [s for s in clip_scores if not np.isnan(s)]
        mean_clip = float(np.mean(valid_clips)) if valid_clips else float("nan")

    # SSIM (original image vs. mitigated image)
    mean_ssim = float("nan")
    ski = _try_import("skimage.metrics", "scikit-image")
    if ski:
        from skimage.metrics import structural_similarity as ssim
        ssim_rows = [r for r in hateful_rows if r.get("mitigated_path")]
        ssim_scores = []
        for r in ssim_rows:
            try:
                orig = np.array(Image.open(img_root / r["img"]).convert("RGB"))
                mit  = np.array(Image.open(r["mitigated_path"]).convert("RGB").resize(
                    (orig.shape[1], orig.shape[0]), Image.LANCZOS))
                ssim_scores.append(ssim(orig, mit, channel_axis=2, data_range=255))
            except Exception:
                ssim_scores.append(float("nan"))
        valid_ssim = [s for s in ssim_scores if not np.isnan(s)]
        mean_ssim = float(np.mean(valid_ssim)) if valid_ssim else float("nan")

    # MPS — Multimodal Preservation Score
    # cosine_sim( norm(CLIP_img_orig + CLIP_txt_orig), norm(CLIP_img_mit + CLIP_txt_mit) )
    mean_mps = float("nan")
    mps_rows = [r for r in hateful_rows
                if r.get("original_text") and r.get("replacement_text") and r.get("mitigated_path")]
    if tf and mps_rows:
        mps_model = tf.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        mps_proc  = tf.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        mps_model.eval()
        mps_scores = []
        for r in mps_rows:
            try:
                orig_img = Image.open(img_root / r["img"]).convert("RGB")
                mit_img  = Image.open(r["mitigated_path"]).convert("RGB")
                with torch.no_grad():
                    def _img_feat(im):
                        inp = mps_proc(images=[im], return_tensors="pt")
                        e = mps_model.get_image_features(**inp)
                        return e / e.norm(dim=-1, keepdim=True)
                    def _txt_feat(t):
                        inp = mps_proc(text=[t], return_tensors="pt", padding=True, truncation=True)
                        e = mps_model.get_text_features(**inp)
                        return e / e.norm(dim=-1, keepdim=True)
                    A = _img_feat(orig_img) + _txt_feat(r["original_text"].replace("\\n", " "))
                    B = _img_feat(mit_img)  + _txt_feat(r["replacement_text"].replace("\\n", " "))
                    A = A / A.norm(dim=-1, keepdim=True)
                    B = B / B.norm(dim=-1, keepdim=True)
                    mps_scores.append((A * B).sum().item())
            except Exception:
                mps_scores.append(float("nan"))
        valid_mps = [s for s in mps_scores if not np.isnan(s)]
        mean_mps = float(np.mean(valid_mps)) if valid_mps else float("nan")

    # --- Timing ---
    t_prompt    = [float(r["t_prompt_s"])    for r in hateful_rows if r.get("t_prompt_s")]
    t_diffusion = [float(r["t_diffusion_s"]) for r in hateful_rows if r.get("t_diffusion_s")]
    tp_arr = np.array(t_prompt)    if t_prompt    else np.array([float("nan")])
    td_arr = np.array(t_diffusion) if t_diffusion else np.array([float("nan")])

    sep = "=" * 52
    print(f"\n{sep}\nMITIGATION METRICS\n{sep}", file=sys.stderr)
    print(f"  N mitigated      : {len(hateful_rows)}")
    if tr_rows:
        print(f"  Mean prob_before : {prob_before.mean():.4f}")
    print(f"  Mean prob_after  : {prob_after.mean():.4f}")
    if tr_per_img.size:
        print(f"  Mean TR%         : {tr_per_img.mean()*100:.1f}%")
    print(f"  % non-hateful    : {pct_nonhateful:.1f}%")
    print(f"  BERTScore F1     : {mean_bert:.4f}" if not np.isnan(mean_bert) else "  BERTScore F1     : n/a")
    print(f"  CLIPScore        : {mean_clip:.4f}" if not np.isnan(mean_clip) else "  CLIPScore        : n/a")
    print(f"  SSIM             : {mean_ssim:.4f}" if not np.isnan(mean_ssim) else "  SSIM             : n/a")
    print(f"  MPS              : {mean_mps:.4f}"  if not np.isnan(mean_mps)  else "  MPS              : n/a")
    print(f"  Prompt gen time  : {tp_arr.mean():.2f}s ± {tp_arr.std():.2f}s")
    print(f"  Diffusion time   : {td_arr.mean():.2f}s ± {td_arr.std():.2f}s")
    print(sep)

    def _r(v): return round(float(v), 4) if not np.isnan(float(v)) else ""

    return {
        "n_mitigated":          len(hateful_rows),
        "mean_prob_before":     _r(prob_before.mean()) if tr_rows else "",
        "mean_prob_after":      _r(prob_after.mean()),
        "mean_tr_pct":          _r(tr_per_img.mean() * 100) if tr_per_img.size else "",
        "pct_nonhateful_after": _r(pct_nonhateful),
        "mean_bertscore_f1":    _r(mean_bert),
        "mean_clip_score":      _r(mean_clip),
        "mean_ssim":            _r(mean_ssim),
        "mean_mps":             _r(mean_mps),
        "prompt_mean_s":        _r(tp_arr.mean()),
        "prompt_std_s":         _r(tp_arr.std()),
        "diffusion_mean_s":     _r(td_arr.mean()),
        "diffusion_std_s":   _r(td_arr.std()),
        "mit_total_mean_s":  _r((tp_arr + td_arr).mean()),
    }


# ============================================================
# STEP 6 — Save summary row
# Merges detection + mitigation metrics into a single dict and
# appends it to <output_root>/summary.csv.
# One row per model — if the model slug already appears in the
# summary, this row is skipped to avoid duplicates.
# ============================================================

def _summary_row_exists(summary_path: Path, eval_type: str, model_slug: str, rep: int) -> bool:
    """Check if a summary row for (eval_type, model_slug, rep) already exists."""
    if not summary_path.exists():
        return False
    key = (eval_type, model_slug, str(rep))
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            row_key = (r.get("eval_type", ""), r.get("model_slug", ""), r.get("rep", ""))
            if row_key == key:
                return True
    return False


def save_summary_row(summary_path: Path, row: dict) -> None:
    # Check for existing entry (composite key: eval_type, model_slug, rep)
    key = (row["eval_type"], row["model_slug"], str(row["rep"]))
    if _summary_row_exists(summary_path, row["eval_type"], row["model_slug"], row["rep"]):
        print(f"[step6] Summary row for {key} already exists — skipping.",
              file=sys.stderr)
        return

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[step6] Summary saved → {summary_path}", file=sys.stderr)


# ============================================================
# AGGREGATION — compute mean ± std across repetitions
# ============================================================

def aggregate_results(summary_path: Path) -> None:
    """
    Read summary.csv, group by (eval_type, vlm_name), compute mean ± std
    for all numeric columns across reps, write summary_aggregated.csv,
    and print a formatted table to stdout.
    """
    if not summary_path.exists():
        print(f"[aggregate] No summary at {summary_path}", file=sys.stderr)
        return

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[aggregate] Summary is empty.", file=sys.stderr)
        return

    # Identify numeric columns: all SUMMARY_FIELDNAMES except the identity columns
    identity_cols = {"eval_type", "rep", "vlm_name", "model_slug"}
    numeric_cols = [c for c in SUMMARY_FIELDNAMES if c not in identity_cols]

    # Group rows by (eval_type, vlm_name)
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["eval_type"], row["vlm_name"])
        groups[key].append(row)

    # Build aggregated rows
    agg_fieldnames = ["eval_type", "vlm_name", "n_reps"]
    for col in numeric_cols:
        agg_fieldnames.append(f"mean_{col}")
        agg_fieldnames.append(f"std_{col}")

    agg_rows = []
    for (eval_type, vlm_name), grp in sorted(groups.items()):
        agg = {"eval_type": eval_type, "vlm_name": vlm_name, "n_reps": len(grp)}
        for col in numeric_cols:
            vals = []
            for r in grp:
                v = r.get(col, "")
                if v != "" and v is not None:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if vals:
                arr = np.array(vals)
                agg[f"mean_{col}"] = round(float(arr.mean()), 4)
                agg[f"std_{col}"]  = round(float(arr.std()), 4)
            else:
                agg[f"mean_{col}"] = ""
                agg[f"std_{col}"]  = ""
        agg_rows.append(agg)

    # Write aggregated CSV
    agg_path = summary_path.parent / "summary_aggregated.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"[aggregate] Aggregated summary → {agg_path}", file=sys.stderr)

    # Print summary table to stdout — key metrics only
    key_metrics = ["auroc", "macro_f1", "accuracy",
                   "mean_bertscore_f1", "mean_clip_score", "mean_ssim", "mean_mps"]
    header_cols = ["eval_type", "vlm_name", "n_reps"] + key_metrics
    col_w = 20

    print("\n" + "=" * (col_w * len(header_cols)), file=sys.stderr)
    header = "  ".join(c[:col_w].ljust(col_w) for c in header_cols)
    print(header, file=sys.stderr)
    print("=" * (col_w * len(header_cols)), file=sys.stderr)

    for agg in agg_rows:
        row_str = []
        row_str.append(str(agg.get("eval_type", ""))[:col_w].ljust(col_w))
        row_str.append(str(agg.get("vlm_name", ""))[:col_w].ljust(col_w))
        row_str.append(str(agg.get("n_reps", "")).ljust(col_w))
        for m in key_metrics:
            mean_v = agg.get(f"mean_{m}", "")
            std_v  = agg.get(f"std_{m}", "")
            if mean_v != "" and std_v != "":
                cell = f"{mean_v:.4f}±{std_v:.4f}"
            else:
                cell = "n/a"
            row_str.append(cell[:col_w].ljust(col_w))
        print("  ".join(row_str), file=sys.stderr)

    print("=" * (col_w * len(header_cols)), file=sys.stderr)


# ============================================================
# MAIN LOOP — iterate over all experiments, models, and repetitions
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-experiment grid evaluation (N reps, multiple eval types)")
    parser.add_argument("--skip_mitigation", action="store_true",
                        help="Run detection only, skip mitigation phases")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip all inference, recompute metrics from existing CSVs")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip all inference, just re-aggregate summary.csv")
    parser.add_argument("--eval_types", nargs="+", default=None,
                        help="Filter to specific eval types (baseline cls_head lora_detect lora_mitigate)")
    parser.add_argument("--reps", type=int, default=None,
                        help="Override N_REPS constant")
    args = parser.parse_args()

    jsonl     = Path(JSONL_PATH)
    img_root  = Path(IMG_DIR)
    out_root  = Path(OUTPUT_ROOT)
    summary_path = out_root / "summary.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load samples once
    samples = load_jsonl(jsonl)
    print(f"Loaded {len(samples)} samples from {jsonl}", file=sys.stderr)

    n_reps = args.reps if args.reps is not None else N_REPS
    experiments = {k: v for k, v in EXPERIMENTS.items()
                   if args.eval_types is None or k in args.eval_types}

    if args.aggregate_only:
        aggregate_results(summary_path)
        return

    # ====================================================================
    # Main experiment loop: eval_type → models → repetitions
    # ====================================================================
    for eval_type, exp_cfg in experiments.items():
        for model_cfg in exp_cfg["models"]:
            vlm_name = model_cfg["vlm_name"]
            slug = slugify(vlm_name)

            # Early skip if all reps already done
            all_done = all(_summary_row_exists(summary_path, eval_type, slug, r) for r in range(n_reps))
            if all_done:
                print(f"\n[skip] {eval_type}/{slug} — all {n_reps} reps already in summary", file=sys.stderr)
                continue

            sep = "=" * 70
            print(f"\n{sep}", file=sys.stderr)
            print(f"EXPERIMENT: {eval_type} | MODEL: {vlm_name}", file=sys.stderr)
            print(sep, file=sys.stderr)

            # ================================================================
            # PHASE A — VLM only: Detection + Mitigation prompt generation
            # ================================================================
            print(f"\n[Phase A] VLM: detection + prompt generation ({n_reps} reps)", file=sys.stderr)

            vlm = processor = cls_head = None
            if not args.metrics_only:
                print("[load] Loading VLM …", file=sys.stderr)
                vlm, processor = instantiate_vlm(
                    vlm_name, CACHE_DIR,
                    adapter_path=model_cfg.get("adapter_path"),
                    mitigation_adapter_path=model_cfg.get("mitigation_adapter_path"),
                )
                print(f"[load] VLM on {vlm.device}", file=sys.stderr)

                if model_cfg.get("cls_head_path"):
                    cls_head = load_cls_head(vlm, model_cfg["cls_head_path"])

            det_metrics_dict = {}  # same across reps (or close)

            for rep in range(n_reps):
                if _summary_row_exists(summary_path, eval_type, slug, rep):
                    print(f"  [rep_{rep:02d}] already done — skip", file=sys.stderr)
                    continue

                rep_dir = out_root / f"{eval_type}_{slug}" / f"rep_{rep:02d}"
                det_csv = rep_dir / "detection_predictions.csv"
                mit_prompt_dir = rep_dir / "mitigated"

                print(f"  [rep_{rep:02d}] Detection + Prompt gen", file=sys.stderr)

                # Detection inference
                if not args.metrics_only:
                    run_detection(vlm, processor, samples, img_root, det_csv, cls_head=cls_head)

                # Detection metrics (once per model, reused across reps)
                if rep == 0 and det_csv.exists():
                    det_metrics_dict = compute_detection_metrics(det_csv)

                # Mitigation prompt generation (if needed)
                if exp_cfg.get("run_mitigation") and not args.skip_mitigation and not args.metrics_only:
                    run_mitigation_prompts(vlm, processor, samples, img_root, det_csv, mit_prompt_dir)

            # Unload VLM
            if vlm is not None:
                del vlm, processor
                if cls_head is not None:
                    del cls_head
                torch.cuda.empty_cache()

            # ================================================================
            # PHASE B — Diffusion only: Image mitigation
            # ================================================================
            if exp_cfg.get("run_mitigation") and not args.skip_mitigation and not args.metrics_only:
                print(f"\n[Phase B] Diffusion: image mitigation ({n_reps} reps)", file=sys.stderr)

                diffusion_model = instantiate_diffusion(DIFFUSION_MODEL, cache_dir=CACHE_DIR)
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

                for rep in range(n_reps):
                    if _summary_row_exists(summary_path, eval_type, slug, rep):
                        print(f"  [rep_{rep:02d}] already done — skip", file=sys.stderr)
                        continue

                    rep_dir = out_root / f"{eval_type}_{slug}" / f"rep_{rep:02d}"
                    mit_csv = rep_dir / "mitigation_results.csv"
                    mit_dir = rep_dir / "mitigated"
                    det_csv = rep_dir / "detection_predictions.csv"

                    print(f"  [rep_{rep:02d}] Diffusion inference", file=sys.stderr)
                    run_mitigation_diffusion(diffusion_model, samples, img_root, mit_csv, mit_dir, generator=generator, det_path=det_csv)

                # Unload diffusion
                del diffusion_model
                torch.cuda.empty_cache()

            # ================================================================
            # PHASE C — VLM only: Judge + Metrics + Save summary
            # ================================================================
            if exp_cfg.get("run_mitigation") and not args.skip_mitigation:
                print(f"\n[Phase C] VLM: judge + metrics ({n_reps} reps)", file=sys.stderr)

                vlm = processor = None
                if not args.metrics_only:
                    print("[load] Loading VLM for judge …", file=sys.stderr)
                    vlm, processor = instantiate_vlm(
                        vlm_name, CACHE_DIR,
                        adapter_path=model_cfg.get("adapter_path"),
                        mitigation_adapter_path=model_cfg.get("mitigation_adapter_path"),
                    )

                for rep in range(n_reps):
                    if _summary_row_exists(summary_path, eval_type, slug, rep):
                        continue

                    rep_dir = out_root / f"{eval_type}_{slug}" / f"rep_{rep:02d}"
                    mit_csv = rep_dir / "mitigation_results.csv"
                    mit_dir = rep_dir / "mitigated"

                    # Judge (VLM on mitigated images)
                    if vlm is not None:
                        run_judge(vlm, processor, mit_csv, mit_dir)

                    # Compute metrics
                    mit_metrics = compute_mitigation_metrics(mit_csv, img_root) if mit_csv.exists() else {}

                    # Save summary row
                    summary_row = {
                        "eval_type": eval_type,
                        "rep": rep,
                        "vlm_name": vlm_name,
                        "model_slug": slug,
                    }
                    summary_row.update(det_metrics_dict)
                    summary_row.update(mit_metrics)
                    save_summary_row(summary_path, summary_row)
                    print(f"  [rep_{rep:02d}] Summary saved", file=sys.stderr)

                # Unload VLM
                if vlm is not None:
                    del vlm, processor
                    torch.cuda.empty_cache()

            else:
                # No mitigation: just save detection metrics for each rep
                print(f"\n[Phase C] Saving summary ({n_reps} reps)", file=sys.stderr)
                for rep in range(n_reps):
                    if not _summary_row_exists(summary_path, eval_type, slug, rep):
                        summary_row = {
                            "eval_type": eval_type,
                            "rep": rep,
                            "vlm_name": vlm_name,
                            "model_slug": slug,
                        }
                        summary_row.update(det_metrics_dict)
                        save_summary_row(summary_path, summary_row)
                        print(f"  [rep_{rep:02d}] Summary saved", file=sys.stderr)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"All experiments done. Summary → {summary_path}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    # Aggregate results
    aggregate_results(summary_path)


if __name__ == "__main__":
    main()

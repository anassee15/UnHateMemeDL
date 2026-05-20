#!/usr/bin/env python3
"""
run_experiment_grid.py — Flexible multi-experiment grid evaluation with YAML config.

Configuration is loaded from a YAML file (e.g., experiments.yaml). The script supports
two-phase mitigation optimization (VLM + diffusion never loaded simultaneously) and
N repetitions per experiment for statistical significance.

Usage
-----
  # Run all experiments from config file
  python src/eval/run_experiment_grid.py --config experiments.yaml

  # Override n_reps and filter eval types
  python src/eval/run_experiment_grid.py --config experiments.yaml --n_reps 5 --eval_types baseline cls_head

  # Detection only (skip mitigation)
  python src/eval/run_experiment_grid.py --config experiments.yaml --skip_mitigation

  # Recompute metrics (no GPU)
  python src/eval/run_experiment_grid.py --config experiments.yaml --metrics_only

  # Aggregate results
  python src/eval/run_experiment_grid.py --config experiments.yaml --aggregate_only
"""

import sys
import json
import csv
import time
import argparse
import re
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

from vlm import instantiate_vlm, detect_hateful_meme, detect_hateful_meme_cls_head, get_diffusion_prompt, load_cls_head
from diffusion import instantiate_diffusion, mitigate_image
from utils import parse_hateful_response, parse_prompt_generation

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

JSONL_PATH   = "data/eval_data/eval_490_balanced.jsonl"
IMG_DIR      = "data/hateful-meme"   # Images root (JSONLs reference img/NNNN.png relative to this)
OUTPUT_ROOT  = "report/full_eval"
CACHE_DIR    = None
DIFFUSION_MODEL = "black-forest-labs/FLUX.2-klein-9B"

DET_FIELDNAMES = [
    "id", "img", "label_true", "text",
    "prob_pred", "label_pred", "classification", "description",
    "modality_type", "error", "t_detect_s",
]

MIT_FIELDNAMES = [
    "id", "img", "label_true",
    "prob_before", "prob_after",
    "detoxify_before", "detoxify_after",
    "hate_location", "severity",
    "original_text", "replacement_text", "flux_prompt",
    "bertscore_f1", "clip_score", "ssim", "mps",
    "mitigated_path", "error",
    "t_prompt_s", "t_diffusion_s",
]

SUMMARY_FIELDNAMES = [
    "eval_type", "rep", "vlm_name", "model_slug",
    "n_total", "n_hateful", "n_nonhateful", "n_errors",
    "auroc", "macro_f1", "accuracy",
    "precision_hateful", "recall_hateful", "f1_hateful",
    "tp", "tn", "fp", "fn",
    "det_mean_s", "det_std_s", "det_median_s", "det_total_s",
    "n_mitigated",
    "mean_prob_before", "mean_prob_after", "mean_tr_pct", "pct_nonhateful_after",
    "mean_bertscore_f1", "mean_clip_score", "mean_ssim", "mean_mps",
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


# ============================================================
# FUNCTIONS IMPORTED FROM run_full_eval.py
# (Copied here for modularity; could be refactored into shared module)
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
                is_hateful, prob = detect_hateful_meme_cls_head(vlm, processor, cls_head, img_path)
                row["t_detect_s"] = f"{time.perf_counter() - t0:.3f}"
                row["prob_pred"]      = prob
                row["label_pred"]     = 1 if prob >= 0.5 else 0
                row["classification"] = "hateful" if is_hateful else "non-hateful"
                row["description"]    = ""
                print(f"prob={prob:.3f} t={row['t_detect_s']}s (cls_head)", file=sys.stderr)
            else:
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


def run_mitigation_prompts(
    vlm, processor,
    samples: list[dict], img_root: Path, det_path: Path,
    out_dir: Path,
) -> None:
    det: dict = {}
    if det_path.exists():
        with open(det_path, newline="") as f:
            det = {int(r["id"]): r for r in csv.DictReader(f)}

    hateful = [s for s in samples if int(s["label"]) == 1]
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
                jpath.write_text(json.dumps({"skipped": "prob < 0.2", "prob": prob_before}))
                print(f"  [{i+1}/{len(todo)}] {img_path.name}: prob={prob_before:.3f} < 0.2 → skip", file=sys.stderr)
                continue

            raw_prompt = get_diffusion_prompt(vlm, processor, img_path)
            mitigation = parse_prompt_generation(raw_prompt)
            jpath.write_text(json.dumps(mitigation, ensure_ascii=False, indent=2))
            print(f"  [{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

        except Exception as e:
            jpath.write_text(json.dumps({"error": str(e)}))
            print(f"  [{i+1}/{len(todo)}] {img_path.name}: ERROR: {e}", file=sys.stderr)

    print(f"[step3a] Prompt generation complete → {out_dir}", file=sys.stderr)


def run_mitigation_diffusion(
    diffusion_model,
    samples: list[dict], img_root: Path,
    mit_csv_path: Path, out_dir: Path,
    generator: torch.Generator,
) -> None:
    hateful = [s for s in samples if int(s["label"]) == 1]
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

        try:
            if not jpath.exists():
                row["error"] = "intermediate.json not found"
                append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)
                print(f"    ERROR: intermediate.json not found", file=sys.stderr)
                continue

            mitigation_data = json.loads(jpath.read_text())

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

            image = Image.open(img_path).convert("RGB")
            t_d = time.perf_counter()
            mitigated = mitigate_image(diffusion_model, image, mitigation_data, generator=generator)
            row["t_diffusion_s"] = f"{time.perf_counter() - t_d:.3f}"
            mitigated.save(mit_path)

            row["hate_location"]    = mitigation_data.get("hate_location", "")
            row["severity"]         = mitigation_data.get("severity", "")
            row["original_text"]    = (mitigation_data.get("original_text") or "").replace("\n", "\\n")
            row["replacement_text"] = (mitigation_data.get("replacement_text") or "").replace("\n", "\\n")
            row["flux_prompt"]      = mitigation_data.get("flux_prompt", "")
            row["t_prompt_s"]       = "0"

            print(f"    diffusion={row['t_diffusion_s']}s", file=sys.stderr)

        except Exception as e:
            row["error"] = str(e)
            print(f"    ERROR: {e}", file=sys.stderr)

        append_csv_row(mit_csv_path, MIT_FIELDNAMES, row)

    print(f"[step3b] Diffusion complete → {out_dir}", file=sys.stderr)


def run_judge(vlm, processor, mit_csv_path: Path, out_dir: Path) -> None:
    if not mit_csv_path.exists():
        print("[step4] No mitigation CSV found — skipping judge.", file=sys.stderr)
        return

    with open(mit_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

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

    mit_csv_path.write_text("")
    with open(mit_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MIT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MIT_FIELDNAMES})

    print(f"[step4] Judge complete → {mit_csv_path}", file=sys.stderr)


def compute_mitigation_metrics(mit_csv_path: Path, img_root: Path) -> dict:
    if not mit_csv_path.exists():
        return {}

    with open(mit_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    hateful_rows = [r for r in rows
                    if r.get("prob_after") and r.get("prob_before")
                    and not r.get("error")
                    and int(r.get("label_true", 0)) == 1]

    if not hateful_rows:
        print("[step5] No valid hateful rows in mitigation CSV.", file=sys.stderr)
        return {}

    prob_before = np.array([float(r["prob_before"]) for r in hateful_rows if r.get("prob_before")], dtype=float)
    prob_after  = np.array([float(r["prob_after"])  for r in hateful_rows], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tr_per_img = np.where(prob_before > 0, (prob_before - prob_after) / prob_before, 0.0)
    pct_nonhateful = (prob_after < 0.5).mean() * 100

    text_rows = [r for r in hateful_rows if r.get("original_text") and r.get("replacement_text")]
    orig_texts = [r["original_text"].replace("\\n", " ") for r in text_rows]
    repl_texts = [r["replacement_text"].replace("\\n", " ") for r in text_rows]

    mean_bert = float("nan")
    bs = _try_import("bert_score")
    if bs and text_rows:
        _, _, F = bs.score(repl_texts, orig_texts, lang="en", verbose=False)
        mean_bert = float(F.mean())

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

    t_prompt    = [float(r["t_prompt_s"])    for r in hateful_rows if r.get("t_prompt_s")]
    t_diffusion = [float(r["t_diffusion_s"]) for r in hateful_rows if r.get("t_diffusion_s")]
    tp_arr = np.array(t_prompt)    if t_prompt    else np.array([float("nan")])
    td_arr = np.array(t_diffusion) if t_diffusion else np.array([float("nan")])

    sep = "=" * 52
    print(f"\n{sep}\nMITIGATION METRICS\n{sep}", file=sys.stderr)
    print(f"  N mitigated      : {len(hateful_rows)}")
    print(f"  Mean prob_before : {prob_before.mean():.4f}")
    print(f"  Mean prob_after  : {prob_after.mean():.4f}")
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
        "n_mitigated":       len(hateful_rows),
        "mean_prob_before":  _r(prob_before.mean()),
        "mean_prob_after":   _r(prob_after.mean()),
        "mean_tr_pct":       _r(tr_per_img.mean() * 100),
        "pct_nonhateful_after": _r(pct_nonhateful),
        "mean_bertscore_f1": _r(mean_bert),
        "mean_clip_score":   _r(mean_clip),
        "mean_ssim":         _r(mean_ssim),
        "mean_mps":          _r(mean_mps),
        "prompt_mean_s":     _r(tp_arr.mean()),
        "prompt_std_s":      _r(tp_arr.std()),
        "diffusion_mean_s":  _r(td_arr.mean()),
        "diffusion_std_s":   _r(td_arr.std()),
        "mit_total_mean_s":  _r((tp_arr + td_arr).mean()),
    }


def save_summary_row(summary_path: Path, row: dict) -> None:
    if _summary_row_exists(summary_path, row["eval_type"], row["model_slug"], row["rep"]):
        key = (row["eval_type"], row["model_slug"], str(row["rep"]))
        print(f"[step6] Summary row for {key} already exists — skipping.", file=sys.stderr)
        return

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[step6] Summary saved → {summary_path}", file=sys.stderr)


def aggregate_results(summary_path: Path) -> None:
    """Compute mean ± std across repetitions, write summary_aggregated.csv, print table."""
    if not summary_path.exists():
        print(f"[aggregate] No summary at {summary_path}", file=sys.stderr)
        return

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[aggregate] Summary is empty.", file=sys.stderr)
        return

    identity_cols = {"eval_type", "rep", "vlm_name", "model_slug"}
    numeric_cols = [c for c in SUMMARY_FIELDNAMES if c not in identity_cols]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["eval_type"], row["vlm_name"])
        groups[key].append(row)

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

    agg_path = summary_path.parent / "summary_aggregated.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"[aggregate] Aggregated summary → {agg_path}", file=sys.stderr)

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
# MAIN LOOP
# ============================================================

def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def normalize_config(config: dict) -> dict:
    """Convert YAML config to internal EXPERIMENTS format."""
    experiments = {}
    for eval_type, exp_cfg in config.get("experiments", {}).items():
        models = []
        for m in exp_cfg.get("models", []):
            if isinstance(m, str):
                # Simple string: just the model name
                models.append({
                    "vlm_name": m,
                    "adapter_path": None,
                    "mitigation_adapter_path": None,
                    "cls_head_path": None,
                })
            elif isinstance(m, dict):
                # Dict with model + optional paths
                models.append({
                    "vlm_name": m.get("model") or m.get("vlm_name"),
                    "adapter_path": m.get("adapter_path"),
                    "mitigation_adapter_path": m.get("mitigation_adapter_path"),
                    "cls_head_path": m.get("cls_head_path"),
                })
        experiments[eval_type] = {
            "run_mitigation": exp_cfg.get("run_mitigation", False),
            "models": models,
        }
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-experiment grid evaluation from YAML config")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML configuration file (e.g., experiments.yaml)")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="Path to evaluation JSONL (default: data/eval_data/eval_490_balanced.jsonl)")
    parser.add_argument("--img_dir", type=Path, default=None,
                        help="Path to images root directory (default: data/hateful-meme)")
    parser.add_argument("--output_root", type=Path, default=None,
                        help="Path to output directory (default: report/full_eval)")
    parser.add_argument("--n_reps", type=int, default=None,
                        help="Override n_reps from config")
    parser.add_argument("--skip_mitigation", action="store_true",
                        help="Run detection only, skip mitigation phases")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip all inference, recompute metrics from existing CSVs")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip all inference, just re-aggregate summary.csv")
    parser.add_argument("--eval_types", nargs="+", default=None,
                        help="Filter to specific eval types")
    parser.add_argument("--hateful_only", action="store_true",
                        help="Test on hateful images only (label=1) for mitigation quality evaluation")
    parser.add_argument("--mitigation_only", action="store_true",
                        help="Skip detection, test mitigation pipeline only (prompt gen → diffusion → judge)")
    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    experiments = normalize_config(config)

    # Use CLI args if provided, otherwise use defaults
    jsonl = Path(args.jsonl) if args.jsonl else Path(JSONL_PATH)
    img_root = Path(args.img_dir) if args.img_dir else Path(IMG_DIR)
    out_root = Path(args.output_root) if args.output_root else Path(OUTPUT_ROOT)
    summary_path = out_root / "summary.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(jsonl)
    print(f"Loaded {len(samples)} samples from {jsonl}", file=sys.stderr)

    # Filter to hateful images only if requested
    if args.hateful_only:
        original_count = len(samples)
        samples = [s for s in samples if int(s.get("label", 0)) == 1]
        print(f"[hateful_only] Filtered: {original_count} → {len(samples)} hateful images", file=sys.stderr)

    n_reps = args.n_reps if args.n_reps is not None else config.get("n_reps", 3)
    experiments = {k: v for k, v in experiments.items()
                   if args.eval_types is None or k in args.eval_types}

    if args.aggregate_only:
        aggregate_results(summary_path)
        return

    # ====================================================================
    # Main experiment loop
    # ====================================================================
    for eval_type, exp_cfg in experiments.items():
        for model_cfg in exp_cfg["models"]:
            vlm_name = model_cfg["vlm_name"]
            slug = slugify(vlm_name)

            all_done = all(_summary_row_exists(summary_path, eval_type, slug, r) for r in range(n_reps))
            if all_done:
                print(f"\n[skip] {eval_type}/{slug} — all {n_reps} reps already in summary", file=sys.stderr)
                continue

            sep = "=" * 70
            print(f"\n{sep}", file=sys.stderr)
            print(f"EXPERIMENT: {eval_type} | MODEL: {vlm_name}", file=sys.stderr)
            print(sep, file=sys.stderr)

            # Phase A: VLM (detection + prompts) — skipped if --mitigation_only or exp config says mitigation_only
            skip_detection = args.mitigation_only or exp_cfg.get("mitigation_only", False)

            if not skip_detection:
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

                det_metrics_dict = {}

                for rep in range(n_reps):
                    if _summary_row_exists(summary_path, eval_type, slug, rep):
                        print(f"  [rep_{rep:02d}] already done — skip", file=sys.stderr)
                        continue

                    rep_dir = out_root / f"{eval_type}_{slug}" / f"rep_{rep:02d}"
                    det_csv = rep_dir / "detection_predictions.csv"
                    mit_prompt_dir = rep_dir / "mitigated"

                    print(f"  [rep_{rep:02d}] Detection + Prompt gen", file=sys.stderr)

                    if not args.metrics_only:
                        run_detection(vlm, processor, samples, img_root, det_csv, cls_head=cls_head)

                    if rep == 0 and det_csv.exists():
                        det_metrics_dict = compute_detection_metrics(det_csv)

                    if exp_cfg.get("run_mitigation") and not args.skip_mitigation and not args.metrics_only:
                        run_mitigation_prompts(vlm, processor, samples, img_root, det_csv, mit_prompt_dir)

                if vlm is not None:
                    del vlm, processor
                    if cls_head is not None:
                        del cls_head
                    torch.cuda.empty_cache()

            else:
                # Skip detection (via --mitigation_only flag or YAML config): go directly to prompt gen + diffusion
                mode_str = "YAML config" if exp_cfg.get("mitigation_only") else "CLI flag"
                print(f"\n[Phase A (mitigation_only via {mode_str})] VLM: prompt generation ({n_reps} reps)", file=sys.stderr)

                vlm = processor = None
                if not args.metrics_only:
                    print("[load] Loading VLM …", file=sys.stderr)
                    vlm, processor = instantiate_vlm(
                        vlm_name, CACHE_DIR,
                        adapter_path=model_cfg.get("adapter_path"),
                        mitigation_adapter_path=model_cfg.get("mitigation_adapter_path"),
                    )
                    print(f"[load] VLM on {vlm.device}", file=sys.stderr)

                det_metrics_dict = {}  # Empty: no detection metrics

                for rep in range(n_reps):
                    if _summary_row_exists(summary_path, eval_type, slug, rep):
                        print(f"  [rep_{rep:02d}] already done — skip", file=sys.stderr)
                        continue

                    rep_dir = out_root / f"{eval_type}_{slug}" / f"rep_{rep:02d}"
                    mit_prompt_dir = rep_dir / "mitigated"

                    print(f"  [rep_{rep:02d}] Prompt generation (skip detection)", file=sys.stderr)

                    # Only generate prompts, skip detection
                    if not args.metrics_only and vlm is not None:
                        run_mitigation_prompts(vlm, processor, samples, img_root, None, mit_prompt_dir)

                if vlm is not None:
                    del vlm, processor
                    torch.cuda.empty_cache()

            # Phase B: Diffusion
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

                    print(f"  [rep_{rep:02d}] Diffusion inference", file=sys.stderr)
                    run_mitigation_diffusion(diffusion_model, samples, img_root, mit_csv, mit_dir, generator=generator)

                del diffusion_model
                torch.cuda.empty_cache()

            # Phase C: VLM judge + summary
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

                    if vlm is not None:
                        run_judge(vlm, processor, mit_csv, mit_dir)

                    mit_metrics = compute_mitigation_metrics(mit_csv, img_root) if mit_csv.exists() else {}

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

                if vlm is not None:
                    del vlm, processor
                    torch.cuda.empty_cache()

            else:
                # No mitigation
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

    aggregate_results(summary_path)


if __name__ == "__main__":
    main()

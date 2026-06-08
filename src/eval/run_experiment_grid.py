#!/usr/bin/env python3
"""
run_experiment_grid.py — Flexible multi-experiment grid evaluation with YAML config.

Configuration is loaded from a YAML file (e.g., experiments.yaml). The VLM and the
diffusion model are never loaded at the same time, and each experiment can be run for
N repetitions.

Stages: detection, prompt, diffusion, judge, metrics. Each reads the previous
stage's artifacts from disk, so stages can run separately. Stage selection per
experiment via a YAML 'stages:' list; --stages overrides it globally. If an
experiment has no 'stages:', they are derived from run_mitigation/mitigation_only.

Usage
-----
  # Whole pipeline, all experiments
  python src/eval/run_experiment_grid.py --config experiments.yaml

  # Detection only
  python src/eval/run_experiment_grid.py --config experiments.yaml --stages detection

  # Diffusion prompt generation only
  python src/eval/run_experiment_grid.py --config experiments.yaml --stages prompt

  # Mitigation (diffusion) only
  python src/eval/run_experiment_grid.py --config experiments.yaml --stages diffusion

  # Prompt generation + mitigation
  python src/eval/run_experiment_grid.py --config experiments.yaml --stages prompt diffusion

  # Detection + prompt generation
  python src/eval/run_experiment_grid.py --config experiments.yaml --stages detection prompt

  # Re-aggregate summary.csv (no GPU)
  python src/eval/run_experiment_grid.py --config experiments.yaml --aggregate_only
"""

import sys
import gc
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

from vlm import (
    instantiate_vlm, detect_hateful_meme, detect_hateful_meme_cls_head,
    get_diffusion_prompt, load_cls_head, run_vlm,
)
from diffusion import instantiate_diffusion, mitigate_image
from utils import parse_hateful_response, parse_prompt_generation
from prompt import ZEROSHOT_DETECTION_PROMPT

# Maps a public --pipeline name to the internal detect_hateful_meme(pipeline=...) value.
# "default" keeps main's adapter-aware behavior; "zeroshot" is handled separately.
_PIPELINE_MAP = {
    "fewshot_synthetic":  "baseline",
    "fewshot_real":       "baseline_v2",
    "sentiment_single":   "single_affect",
    "sentiment_chained":  "affect",
    "category_sentiment": "categorized",
    "category_fewshot":   "categorized_v2",
}

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


JSONL_PATH   = "data/eval_data/eval_490_balanced.jsonl"
IMG_DIR      = "data/hateful-meme"   # Images root (JSONLs reference img/NNNN.png relative to this)
OUTPUT_ROOT  = "report/lora_mitigate"
CACHE_DIR    = "../hf_cache"
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
    "hate_source", "hate_location",
    "original_text", "replacement_text", "diffusion_prompt",
    "bertscore_f1", "clip_score", "ssim",
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
    "mean_bertscore_f1", "mean_clip_score", "mean_ssim",
    "prompt_mean_s", "prompt_std_s",
    "diffusion_mean_s", "diffusion_std_s",
    "mit_total_mean_s",
]


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


def run_detection(vlm, processor, samples: list[dict], img_root: Path, out_path: Path,
                  *, cls_head=None, pipeline=None) -> None:
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
                if pipeline in (None, "default"):
                    raw = detect_hateful_meme(vlm, processor, img_path)
                elif pipeline == "zeroshot":
                    raw = run_vlm(vlm, processor, img_path, ZEROSHOT_DETECTION_PROMPT, temperature=0.95)
                else:
                    raw = detect_hateful_meme(vlm, processor, img_path, pipeline=_PIPELINE_MAP[pipeline])
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

    if not valid:
        print("[step2] No valid rows — skipping detection metrics.", file=sys.stderr)
        return {"n_total": 0, "n_errors": n_errors}

    y_true = np.array([int(r["label_true"]) for r in valid])
    y_prob = np.array([float(r["prob_pred"]) for r in valid])
    y_pred = np.array([int(r["label_pred"]) for r in valid])

    # AUROC is undefined when only one class is present (e.g. --hateful_only)
    auroc    = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc      = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_hat   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    # force a 2x2 matrix so tn/fp/fn/tp unpack even with a single class
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    times = [float(r["t_detect_s"]) for r in valid if r.get("t_detect_s")]
    t_arr = np.array(times) if times else np.array([float("nan")])

    sep = "=" * 52
    print(f"\n{sep}\nDETECTION METRICS\n{sep}", file=sys.stderr)
    print(f"  N={len(valid)}  hateful={y_true.sum()}  non-hateful={(~y_true.astype(bool)).sum()}")
    print(f"  AUROC    : {auroc:.4f}" if not np.isnan(auroc) else "  AUROC    : n/a (single class)")
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
        "auroc":        round(auroc, 4) if not np.isnan(auroc) else "",
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
    if det_path is not None and det_path.exists():
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

            if prob_before < 0.5:
                jpath.write_text(json.dumps({"skipped": "prob < 0.5", "prob": prob_before}))
                print(f"  [{i+1}/{len(todo)}] {img_path.name}: prob={prob_before:.3f} < 0.5 → skip", file=sys.stderr)
                continue

            t_p = time.perf_counter()
            raw_prompt = get_diffusion_prompt(vlm, processor, img_path)
            mitigation = parse_prompt_generation(raw_prompt)
            # carry the prompt-generation time to the diffusion stage via the json
            mitigation["_t_prompt_s"] = round(time.perf_counter() - t_p, 3)
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
    det_path: Path | None = None,
    prompt_dir: Path | None = None,
) -> None:
    prompt_dir = prompt_dir or out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mit_csv_path.parent.mkdir(parents=True, exist_ok=True)
    det: dict = {}
    if det_path is not None and det_path.exists():
        with open(det_path, newline="") as f:
            det = {int(r["id"]): r for r in csv.DictReader(f)}

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
        jpath = prompt_dir / f"{sid}_intermediate.json"

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
                print(f"    Skipped (prob < 0.5)", file=sys.stderr)
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

            row["hate_source"]      = mitigation_data.get("hate_source", "")
            row["hate_location"]    = mitigation_data.get("hate_location", "")
            row["original_text"]    = (mitigation_data.get("original_text") or "").replace("\n", "\\n")
            row["replacement_text"] = (mitigation_data.get("replacement_text") or "").replace("\n", "\\n")
            row["diffusion_prompt"] = mitigation_data.get("diffusion_prompt", "")
            row["t_prompt_s"]       = mitigation_data.get("_t_prompt_s", "")

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


def compute_mitigation_metrics(mit_csv_path: Path, img_root: Path, prob_before_lookup: dict | None = None) -> dict:
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

    prob_after = np.array([float(r["prob_after"]) for r in hateful_rows], dtype=float)
    pct_nonhateful = (prob_after < 0.5).mean() * 100

    # TR%: prob_before is taken (in order) from the row's measured value, then from a
    # source experiment's detection (prob_before_from), else assumed 1.0 (GT-hateful).
    lookup = prob_before_lookup or {}
    def _prob_before(r):
        if r.get("prob_before"):
            return float(r["prob_before"])
        v = lookup.get(int(r["id"]))
        if v:
            return float(v)
        return 1.0
    n_assumed = sum(1 for r in hateful_rows
                    if not r.get("prob_before") and not lookup.get(int(r["id"])))
    if n_assumed:
        print(f"[step5] prob_before assumed 1.0 for {n_assumed}/{len(hateful_rows)} "
              f"GT-hateful rows (no detection available)", file=sys.stderr)
    prob_before = np.array([_prob_before(r) for r in hateful_rows], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tr_per_img = np.where(prob_before > 0, (prob_before - prob_after) / prob_before, 0.0)

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

    # time only the images that actually ran the full mitigation (skip passthrough/errored),
    # keeping prompt/diffusion arrays parallel so the per-image total is well defined
    processed = [r for r in hateful_rows
                 if r.get("t_diffusion_s") and float(r["t_diffusion_s"]) > 0]
    if processed:
        tp_arr = np.array([float(r.get("t_prompt_s") or 0.0) for r in processed])
        td_arr = np.array([float(r["t_diffusion_s"]) for r in processed])
    else:
        tp_arr = td_arr = np.array([float("nan")])

    sep = "=" * 52
    print(f"\n{sep}\nMITIGATION METRICS\n{sep}", file=sys.stderr)
    print(f"  N mitigated      : {len(hateful_rows)}")
    print(f"  Mean prob_before : {prob_before.mean():.4f}")
    print(f"  Mean prob_after  : {prob_after.mean():.4f}")
    if tr_per_img.size:
        print(f"  Mean TR%         : {tr_per_img.mean()*100:.1f}%")
    print(f"  % non-hateful    : {pct_nonhateful:.1f}%")
    print(f"  BERTScore F1     : {mean_bert:.4f}" if not np.isnan(mean_bert) else "  BERTScore F1     : n/a")
    print(f"  CLIPScore        : {mean_clip:.4f}" if not np.isnan(mean_clip) else "  CLIPScore        : n/a")
    print(f"  SSIM             : {mean_ssim:.4f}" if not np.isnan(mean_ssim) else "  SSIM             : n/a")
    print(f"  Prompt gen time  : {tp_arr.mean():.2f}s ± {tp_arr.std():.2f}s")
    print(f"  Diffusion time   : {td_arr.mean():.2f}s ± {td_arr.std():.2f}s")
    print(sep)

    def _r(v): return round(float(v), 4) if not np.isnan(float(v)) else ""

    return {
        "n_mitigated":          len(hateful_rows),
        "mean_prob_before":     _r(prob_before.mean()),
        "mean_prob_after":      _r(prob_after.mean()),
        "mean_tr_pct":          _r(tr_per_img.mean() * 100) if tr_per_img.size else "",
        "pct_nonhateful_after": _r(pct_nonhateful),
        "mean_bertscore_f1":    _r(mean_bert),
        "mean_clip_score":      _r(mean_clip),
        "mean_ssim":            _r(mean_ssim),
        "prompt_mean_s":        _r(tp_arr.mean()),
        "prompt_std_s":         _r(tp_arr.std()),
        "diffusion_mean_s":     _r(td_arr.mean()),
        "diffusion_std_s":      _r(td_arr.std()),
        "mit_total_mean_s":     _r((tp_arr + td_arr).mean()),
    }


def save_summary_row(summary_path: Path, row: dict) -> None:
    # upsert: replace any existing row for the same (eval_type, model_slug, rep)
    key = (row["eval_type"], row["model_slug"], str(row["rep"]))
    existing = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            existing = [r for r in csv.DictReader(f)
                        if (r.get("eval_type"), r.get("model_slug"), r.get("rep")) != key]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    print(f"[metrics] Summary saved → {summary_path}", file=sys.stderr)


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
                   "mean_tr_pct", "pct_nonhateful_after",
                   "mean_bertscore_f1", "mean_clip_score", "mean_ssim",
                   "det_mean_s", "prompt_mean_s", "diffusion_mean_s"]
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
                models.append({"vlm_name": m, "adapter_path": None,
                               "mitigation_adapter_path": None, "cls_head_path": None,
                               "pipeline": None})
            elif isinstance(m, dict):
                models.append({
                    "vlm_name": m.get("model") or m.get("vlm_name"),
                    "adapter_path": m.get("adapter_path"),
                    "mitigation_adapter_path": m.get("mitigation_adapter_path"),
                    "cls_head_path": m.get("cls_head_path"),
                    "pipeline": m.get("pipeline"),
                })
        stages = exp_cfg.get("stages")
        if stages is not None:
            bad = [s for s in stages if s not in STAGES]
            if bad:
                raise ValueError(f"Experiment '{eval_type}': invalid stages {bad}; valid: {STAGES}")
        experiments[eval_type] = {
            "stages": list(stages) if stages is not None else None,
            "run_mitigation": exp_cfg.get("run_mitigation", False),
            "mitigation_only": exp_cfg.get("mitigation_only", False),
            "pipeline": exp_cfg.get("pipeline"),
            "prob_before_from": exp_cfg.get("prob_before_from"),
            "models": models,
        }
    return experiments


STAGES = ["detection", "prompt", "diffusion", "judge", "metrics"]


def load_prob_before(out_root: Path, src_eval_type: str, slug: str, n_reps: int) -> dict:
    """id -> prob_pred from a source experiment's detection CSV (reuse its prob_before)."""
    for rep in range(n_reps):
        det = out_root / f"{src_eval_type}_{slug}" / f"rep_{rep:02d}" / "detection_predictions.csv"
        if det.exists():
            with open(det, newline="") as f:
                return {int(r["id"]): r["prob_pred"] for r in csv.DictReader(f) if r.get("prob_pred")}
    return {}


def needs_judging(base: Path, n_reps: int) -> bool:
    """True if any mitigated image still lacks prob_after (judge not run yet)."""
    for rep in range(n_reps):
        mit_csv = base / f"rep_{rep:02d}" / "mitigation_results.csv"
        if not mit_csv.exists():
            continue
        with open(mit_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("mitigated_path") and not r.get("prob_after") and not r.get("error"):
                    return True
    return False


def resolve_stages(exp_cfg, cli_stages):
    """Effective stages for an experiment: CLI override > YAML 'stages' > legacy flags."""
    if cli_stages is not None:
        return set(cli_stages)
    if exp_cfg.get("stages") is not None:
        return set(exp_cfg["stages"])
    stages = {"detection", "metrics"}
    if exp_cfg.get("run_mitigation"):
        stages |= {"prompt", "diffusion", "judge"}
    if exp_cfg.get("mitigation_only"):
        stages.discard("detection")
    return stages


def load_vlm(model_cfg: dict, cache_dir):
    vlm, processor = instantiate_vlm(
        model_cfg["vlm_name"], cache_dir,
        adapter_path=model_cfg.get("adapter_path"),
        mitigation_adapter_path=model_cfg.get("mitigation_adapter_path"),
    )
    cls_head = load_cls_head(vlm, model_cfg["cls_head_path"]) if model_cfg.get("cls_head_path") else None
    print(f"[load] VLM on {vlm.device}", file=sys.stderr)
    return vlm, processor, cls_head


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiment(eval_type, exp_cfg, model_cfg, samples, img_root, out_root,
                   summary_path, n_reps, cli_stages, cache_dir,
                   diffusion_model, diffusion_offload):
    vlm_name = model_cfg["vlm_name"]
    slug = slugify(vlm_name)
    base = out_root / f"{eval_type}_{slug}"
    prompt_dir = base / "prompts"
    pipeline = model_cfg.get("pipeline") or exp_cfg.get("pipeline")

    stages = resolve_stages(exp_cfg, cli_stages)
    do_detection = "detection" in stages
    do_prompt    = "prompt"    in stages
    do_diffusion = "diffusion" in stages
    do_judge     = "judge"     in stages
    do_metrics   = "metrics"   in stages

    sep = "=" * 70
    print(f"\n{sep}\nEXPERIMENT: {eval_type} | MODEL: {vlm_name} | stages: {sorted(stages)}\n{sep}",
          file=sys.stderr)

    def rep_dir(rep):
        return base / f"rep_{rep:02d}"

    # VLM: detection (+ prompt generation), then free the VLM
    if do_detection or do_prompt:
        vlm, processor, cls_head = load_vlm(model_cfg, cache_dir)
        if do_detection:
            for rep in range(n_reps):
                print(f"[detection] rep {rep}", file=sys.stderr)
                run_detection(vlm, processor, samples, img_root,
                              rep_dir(rep) / "detection_predictions.csv",
                              cls_head=cls_head, pipeline=pipeline)
        if do_prompt:
            det0 = rep_dir(0) / "detection_predictions.csv"
            run_mitigation_prompts(vlm, processor, samples, img_root,
                                   det0 if det0.exists() else None, prompt_dir)
        del vlm, processor, cls_head
        free_gpu()

    # diffusion only, with the full GPU now that the VLM is gone
    if do_diffusion:
        diffusion = instantiate_diffusion(diffusion_model, cache_dir=cache_dir,
                                          offload=diffusion_offload)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for rep in range(n_reps):
            print(f"[diffusion] rep {rep}", file=sys.stderr)
            rd = rep_dir(rep)
            det_csv = rd / "detection_predictions.csv"
            generator = torch.Generator(device=device).manual_seed(42 + rep)
            run_mitigation_diffusion(diffusion, samples, img_root,
                                     rd / "mitigation_results.csv", rd / "mitigated",
                                     generator=generator,
                                     det_path=det_csv if det_csv.exists() else None,
                                     prompt_dir=prompt_dir)
        del diffusion
        free_gpu()

    # auto-run judge before metrics when mitigated images still lack prob_after
    if do_metrics and not do_judge and needs_judging(base, n_reps):
        print("[judge] auto-enabled before metrics (mitigation results missing prob_after)",
              file=sys.stderr)
        do_judge = True

    # VLM judge on the mitigated images
    if do_judge:
        vlm, processor, cls_head = load_vlm(model_cfg, cache_dir)
        for rep in range(n_reps):
            mit_csv = rep_dir(rep) / "mitigation_results.csv"
            if mit_csv.exists():
                print(f"[judge] rep {rep}", file=sys.stderr)
                run_judge(vlm, processor, mit_csv, rep_dir(rep) / "mitigated")
        del vlm, processor, cls_head
        free_gpu()

    # metrics + summary row (upserted, so this is safe to re-run)
    if do_metrics:
        # optionally reuse prob_before from another experiment's detection (e.g. baseline)
        pb_lookup = None
        if exp_cfg.get("prob_before_from"):
            pb_lookup = load_prob_before(out_root, exp_cfg["prob_before_from"], slug, n_reps)
            print(f"[metrics] reusing prob_before from '{exp_cfg['prob_before_from']}' "
                  f"({len(pb_lookup)} ids)", file=sys.stderr)
        for rep in range(n_reps):
            rd = rep_dir(rep)
            det_csv = rd / "detection_predictions.csv"
            mit_csv = rd / "mitigation_results.csv"
            det_metrics = compute_detection_metrics(det_csv) if det_csv.exists() else {}
            mit_metrics = compute_mitigation_metrics(mit_csv, img_root, prob_before_lookup=pb_lookup) if mit_csv.exists() else {}
            if not det_metrics and not mit_metrics:
                continue
            row = {"eval_type": eval_type, "rep": rep, "vlm_name": vlm_name, "model_slug": slug}
            row.update(det_metrics)
            row.update(mit_metrics)
            save_summary_row(summary_path, row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-experiment grid evaluation from YAML config")
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML configuration file (e.g., experiments.yaml)")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--img_dir", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--diffusion_model", default=None,
                        help=f"Diffusion model HF id (default: {DIFFUSION_MODEL})")
    parser.add_argument("--n_reps", type=int, default=None, help="Override n_reps from config")
    parser.add_argument("--stages", nargs="+", default=None, choices=STAGES,
                        help="Stages to run (default: all). Examples: "
                             "detection only '--stages detection'; "
                             "prompt only '--stages prompt'; "
                             "mitigation only '--stages diffusion'; "
                             "prompt + mitigation '--stages prompt diffusion'; "
                             "detection + prompt '--stages detection prompt'.")
    parser.add_argument("--eval_types", nargs="+", default=None, help="Filter to specific eval types")
    parser.add_argument("--hateful_only", action="store_true",
                        help="Evaluate on hateful images only (label=1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N samples (quick smoke test)")
    parser.add_argument("--diffusion_offload", action="store_true",
                        help="Use CPU offload for the diffusion model (low-VRAM GPUs)")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip inference, just re-aggregate summary.csv")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    config = load_config(args.config)
    experiments = normalize_config(config)

    jsonl = Path(args.jsonl) if args.jsonl else Path(JSONL_PATH)
    img_root = Path(args.img_dir) if args.img_dir else Path(IMG_DIR)
    out_root = Path(args.output_root) if args.output_root else Path(OUTPUT_ROOT)
    cache_dir = args.cache_dir if args.cache_dir is not None else CACHE_DIR
    diffusion_model = args.diffusion_model or DIFFUSION_MODEL
    summary_path = out_root / "summary.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(summary_path)
        return

    samples = load_jsonl(jsonl)
    print(f"Loaded {len(samples)} samples from {jsonl}", file=sys.stderr)
    if args.hateful_only:
        n0 = len(samples)
        samples = [s for s in samples if int(s.get("label", 0)) == 1]
        print(f"[hateful_only] {n0} -> {len(samples)} hateful images", file=sys.stderr)
    if args.limit:
        samples = samples[:args.limit]
        print(f"[limit] using first {len(samples)} samples", file=sys.stderr)

    n_reps = args.n_reps if args.n_reps is not None else config.get("n_reps", 3)
    cli_stages = set(args.stages) if args.stages else None
    experiments = {k: v for k, v in experiments.items()
                   if args.eval_types is None or k in args.eval_types}

    stages_src = f"CLI override {sorted(cli_stages)}" if cli_stages else "per-experiment (YAML)"
    print(f"[stages] {stages_src}  n_reps={n_reps}", file=sys.stderr)

    for eval_type, exp_cfg in experiments.items():
        for model_cfg in exp_cfg["models"]:
            run_experiment(eval_type, exp_cfg, model_cfg, samples, img_root, out_root,
                           summary_path, n_reps, cli_stages, cache_dir,
                           diffusion_model, args.diffusion_offload)

    aggregate_results(summary_path)
    print(f"\nDone. Summary -> {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Mitigation evaluation for UnHateMemeDL.

Three phases (each can be run independently):

  Phase 1 (very slow, GPU)  --run_mitigation
      Runs the full pipeline on label=1 images.
      Saves:  <out_dir>/<id>_mitigated.png
              <out_dir>/<id>_intermediate.json
      Resumable: skips images whose output files already exist.

  Phase 2 (slow, GPU)       --run_judge
      Re-runs VLM detection on every mitigated image to get prob_after.
      Saves results to <output_csv>.

  Phase 3 (fast, CPU)       --compute_metrics
      Loads <output_csv> and computes:
        Axis A  — Toxicity Reduction (TR%)
        Axis B  — Content preservation (BERTScore, CLIPScore, SSIM)
        Joint   — % non-hateful AND coherent, Pareto CSV
        Failure — Over-sanitization rate (label=0 with prob_before >= 0.2)

Typical usage
-------------
  # Full run (all 3 phases):
  python src/eval/run_mitigation_eval.py --all \\
      --jsonl      data/eval_data/eval_490_balanced.jsonl \\
      --img_dir    <dataset-root> \\
      --det_csv    report/detection_predictions.csv \\
      --out_dir    report/mitigated \\
      --output_csv report/mitigation_results.csv

  # Metrics only (phases 1 & 2 already done):
  python src/eval/run_mitigation_eval.py --compute_metrics \\
      --det_csv    report/detection_predictions.csv \\
      --output_csv report/mitigation_results.csv

Extra dependencies (not in requirements.txt):
  pip install bert_score detoxify scikit-image
  # CLIPScore uses transformers (already installed)
"""

import sys
import json
import argparse
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

from vlm import instantiate_vlm, detect_hateful_meme, get_diffusion_prompt
from diffusion import instantiate_diffusion, mitigate_image
from utils import parse_hateful_response, parse_prompt_generation

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "id", "img", "label_true",
    "prob_before", "prob_after",
    "detoxify_before", "detoxify_after",
    "hate_location", "severity",
    "original_text", "replacement_text", "flux_prompt",
    "bertscore_f1", "clip_score", "ssim",
    "mitigated_path", "error",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_det_csv(path: Path) -> dict:
    """Load detection results keyed by id."""
    with open(path, newline="") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def load_mit_csv(path: Path) -> dict:
    """Load existing mitigation results keyed by id."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def open_csv_writer(path: Path, append: bool):
    write_header = not path.exists() or path.stat().st_size == 0 or not append
    f = open(path, "a" if append else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
    return f, writer


# ---------------------------------------------------------------------------
# Phase 1 — run mitigation pipeline on hateful images
# ---------------------------------------------------------------------------

def run_mitigation(args):
    samples  = load_jsonl(Path(args.jsonl))
    img_root = Path(args.img_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det      = load_det_csv(Path(args.det_csv))

    # Only process ground-truth hateful images
    hateful = [s for s in samples if int(s["label"]) == 1]
    # Resume: skip those whose output files already exist
    todo = [
        s for s in hateful
        if not (out_dir / f"{s['id']}_mitigated.png").exists()
        or not (out_dir / f"{s['id']}_intermediate.json").exists()
    ]

    if not todo:
        print("[info] All hateful images already mitigated.", file=sys.stderr)
        return

    print(f"[info] {len(hateful) - len(todo)} already done, {len(todo)} remaining.", file=sys.stderr)

    vlm, processor = instantiate_vlm(args.vlm_name, args.cache_dir)
    diffusion_model = instantiate_diffusion(args.diffusion_model_name, cache_dir=args.cache_dir)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    for i, sample in enumerate(todo):
        sid      = sample["id"]
        img_path = img_root / sample["img"]
        print(f"\n[{i+1}/{len(todo)}] {img_path.name}", file=sys.stderr)

        mit_path  = out_dir / f"{sid}_mitigated.png"
        json_path = out_dir / f"{sid}_intermediate.json"

        try:
            image = Image.open(img_path).convert("RGB")

            # Use prob_before from detection CSV when available (avoids redundant VLM call)
            det_row = det.get(int(sid))
            if det_row and det_row.get("prob_pred"):
                prob_before = float(det_row["prob_pred"])
            else:
                raw = detect_hateful_meme(vlm, processor, img_path)
                _, prob_before, _ = parse_hateful_response(raw)

            if prob_before < 0.2:
                # Treat as not hateful — save original unchanged
                image.save(mit_path)
                json_path.write_text(json.dumps({"skipped": "prob < 0.2", "prob": prob_before}))
                print(f"         prob={prob_before:.3f} < 0.2 → saved unchanged", file=sys.stderr)
                continue

            # Generate mitigation plan
            raw_prompt = get_diffusion_prompt(vlm, processor, img_path)
            mitigation = parse_prompt_generation(raw_prompt)
            json_path.write_text(json.dumps(mitigation, ensure_ascii=False, indent=2))

            # Apply mitigation
            mitigated = mitigate_image(diffusion_model, image, mitigation, generator=generator)
            mitigated.save(mit_path)
            print(f"         prob={prob_before:.3f}  loc={mitigation.get('hate_location','?')}  → saved", file=sys.stderr)

        except Exception as e:
            print(f"[error] {img_path.name}: {e}", file=sys.stderr)
            json_path.write_text(json.dumps({"error": str(e)}))

    print(f"\n[info] Mitigation complete → {out_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 2 — VLM judge on mitigated images (get prob_after)
# ---------------------------------------------------------------------------

def run_judge(args):
    samples  = load_jsonl(Path(args.jsonl))
    img_root = Path(args.img_dir)
    out_dir  = Path(args.out_dir)
    det      = load_det_csv(Path(args.det_csv))
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_mit_csv(out_path)

    # Process all images that have a mitigated output
    todo = [
        s for s in samples
        if (out_dir / f"{s['id']}_mitigated.png").exists()
        and int(s["id"]) not in existing
    ]

    if not todo:
        print("[info] All mitigated images already judged.", file=sys.stderr)
        return

    print(f"[info] {len(todo)} images to judge.", file=sys.stderr)

    vlm, processor = instantiate_vlm(args.vlm_name, args.cache_dir)

    f, writer = open_csv_writer(out_path, append=True)
    with f:
        for i, sample in enumerate(todo):
            sid       = sample["id"]
            mit_path  = out_dir / f"{sid}_mitigated.png"
            json_path = out_dir / f"{sid}_intermediate.json"
            det_row   = det.get(int(sid), {})

            print(f"\n[{i+1}/{len(todo)}] judging {mit_path.name}", file=sys.stderr)

            row = {k: "" for k in FIELDNAMES}
            row["id"]          = sid
            row["img"]         = sample["img"]
            row["label_true"]  = sample["label"]
            row["prob_before"] = det_row.get("prob_pred", "")

            # Load intermediate JSON
            if json_path.exists():
                try:
                    mit_json = json.loads(json_path.read_text())
                    row["hate_location"]   = mit_json.get("hate_location", "")
                    row["severity"]        = mit_json.get("severity", "")
                    row["original_text"]   = (mit_json.get("original_text") or "").replace("\n", "\\n")
                    row["replacement_text"] = (mit_json.get("replacement_text") or "").replace("\n", "\\n")
                    row["flux_prompt"]     = mit_json.get("flux_prompt", "")
                except Exception:
                    pass

            # Judge mitigated image
            try:
                raw = detect_hateful_meme(vlm, processor, mit_path)
                _, prob_after, _ = parse_hateful_response(raw)
                row["prob_after"] = prob_after
                print(f"         before={row['prob_before']}  after={prob_after:.3f}", file=sys.stderr)
            except Exception as e:
                print(f"[error] judge failed: {e}", file=sys.stderr)
                row["error"] = str(e)

            row["mitigated_path"] = str(mit_path)
            writer.writerow(row)
            f.flush()

    print(f"\n[info] Judge complete → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 3 — compute metrics
# ---------------------------------------------------------------------------

def _try_import(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[warn] '{pip_name or pkg}' not installed — skipping that metric. "
              f"Install with: pip install {pip_name or pkg}", file=sys.stderr)
        return None


def compute_bertscore(refs: list[str], hyps: list[str]) -> list[float]:
    bs = _try_import("bert_score")
    if bs is None or not refs:
        return []
    _, _, F = bs.score(hyps, refs, lang="en", verbose=False)
    return F.tolist()


def compute_detoxify_scores(texts: list[str]) -> list[float]:
    det = _try_import("detoxify")
    if det is None or not texts:
        return []
    scores = det.Detoxify("original").predict(texts)["toxicity"]
    return list(scores) if hasattr(scores, "__iter__") else [scores]


def compute_clip_scores(image_paths: list[Path], texts: list[str]) -> list[float]:
    if not image_paths:
        return []
    tf = _try_import("transformers")
    if tf is None:
        return []
    model_name = "openai/clip-vit-base-patch32"
    clip_model = tf.CLIPModel.from_pretrained(model_name)
    clip_proc  = tf.CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    scores = []
    for img_path, text in zip(image_paths, texts):
        try:
            image  = Image.open(img_path).convert("RGB")
            inputs = clip_proc(text=[text], images=[image], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = clip_model(**inputs)
            img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.append((img_e * txt_e).sum().item())
        except Exception as e:
            print(f"[warn] CLIPScore failed for {img_path.name}: {e}", file=sys.stderr)
            scores.append(float("nan"))
    return scores


def compute_ssim_scores(orig_paths: list[Path], mit_paths: list[Path]) -> list[float]:
    ski = _try_import("skimage.metrics", "scikit-image")
    if ski is None:
        return []
    from skimage.metrics import structural_similarity as ssim
    scores = []
    for op, mp in zip(orig_paths, mit_paths):
        try:
            orig = np.array(Image.open(op).convert("RGB"))
            mit  = np.array(Image.open(mp).convert("RGB").resize(
                (orig.shape[1], orig.shape[0]), Image.LANCZOS))
            scores.append(ssim(orig, mit, channel_axis=2, data_range=255))
        except Exception as e:
            print(f"[warn] SSIM failed for {op.name}: {e}", file=sys.stderr)
            scores.append(float("nan"))
    return scores


def compute_metrics(args):
    out_path = Path(args.output_csv)
    det_path = Path(args.det_csv)

    if not out_path.exists():
        print(f"[error] No mitigation results at {out_path}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    img_root = Path(args.img_dir) if args.img_dir else None

    valid = [r for r in rows if r.get("prob_after") and not r.get("error")]
    hateful_rows = [r for r in valid if int(r["label_true"]) == 1]

    sep = "=" * 56

    # ------------------------------------------------------------------
    # Axis A — Toxicity Reduction
    # ------------------------------------------------------------------
    prob_before = np.array([float(r["prob_before"]) for r in hateful_rows
                            if r.get("prob_before")], dtype=float)
    prob_after  = np.array([float(r["prob_after"])  for r in hateful_rows], dtype=float)

    # TR% per image, then mean
    with np.errstate(divide="ignore", invalid="ignore"):
        tr_per_image = np.where(prob_before > 0,
                                (prob_before - prob_after) / prob_before,
                                0.0)

    pct_nonhateful = (prob_after < 0.5).mean() * 100

    print(sep)
    print("MITIGATION METRICS — AXIS A: Toxicity Reduction")
    print(sep)
    print(f"  N (hateful images mitigated): {len(hateful_rows)}")
    print(f"  Mean prob_before : {prob_before.mean():.4f}")
    print(f"  Mean prob_after  : {prob_after.mean():.4f}")
    print(f"  TR%  (mean)      : {tr_per_image.mean()*100:.1f}%")
    print(f"  % images prob_after < 0.5 : {pct_nonhateful:.1f}%")

    # Detoxify on text
    text_rows = [r for r in hateful_rows
                 if r.get("original_text") and r.get("replacement_text")]
    if text_rows:
        orig_texts = [r["original_text"].replace("\\n", " ") for r in text_rows]
        repl_texts = [r["replacement_text"].replace("\\n", " ") for r in text_rows]
        det_before = compute_detoxify_scores(orig_texts)
        det_after  = compute_detoxify_scores(repl_texts)
        if det_before and det_after:
            db = np.array(det_before)
            da = np.array(det_after)
            print(f"\n  Detoxify (text, n={len(text_rows)}):")
            print(f"    Before : {db.mean():.4f}")
            print(f"    After  : {da.mean():.4f}")
            print(f"    TR%    : {((db - da) / np.where(db > 0, db, 1)).mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # Axis B — Content Preservation
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("MITIGATION METRICS — AXIS B: Content Preservation")
    print(sep)

    # BERTScore (text pairs only)
    if text_rows:
        bs_scores = compute_bertscore(orig_texts, repl_texts)
        if bs_scores:
            print(f"  BERTScore F1 (n={len(text_rows)}): {np.mean(bs_scores):.4f}")

    # CLIPScore (mitigated image + replacement text)
    clip_rows = [r for r in hateful_rows
                 if r.get("replacement_text") and r.get("mitigated_path")]
    if clip_rows:
        mit_paths = [Path(r["mitigated_path"]) for r in clip_rows]
        clip_texts = [r["replacement_text"].replace("\\n", " ") for r in clip_rows]
        clip_scores = compute_clip_scores(mit_paths, clip_texts)
        if clip_scores:
            arr = np.array([s for s in clip_scores if not np.isnan(s)])
            print(f"  CLIPScore      (n={len(arr)}): {arr.mean():.4f}")

    # SSIM (original vs mitigated)
    if img_root:
        ssim_rows = [r for r in hateful_rows if r.get("mitigated_path")]
        orig_paths = [img_root / r["img"] for r in ssim_rows]
        mit_paths  = [Path(r["mitigated_path"]) for r in ssim_rows]
        ssim_scores = compute_ssim_scores(orig_paths, mit_paths)
        if ssim_scores:
            arr = np.array([s for s in ssim_scores if not np.isnan(s)])
            print(f"  SSIM           (n={len(arr)}): {arr.mean():.4f}")

    # ------------------------------------------------------------------
    # Joint metric
    # ------------------------------------------------------------------
    clip_threshold = 0.20  # cosine similarity threshold for "coherent"
    if clip_rows and clip_scores:
        cs = np.array(clip_scores)
        joint = ((prob_after[:len(cs)] < 0.5) & (cs > clip_threshold)).mean() * 100
        print(f"\n  % non-hateful AND coherent (CLIPScore>{clip_threshold}): {joint:.1f}%")

    # ------------------------------------------------------------------
    # Over-sanitization (label=0)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("FAILURE-MODE: Over-sanitization (label=0 images)")
    print(sep)
    if det_path.exists():
        det = load_det_csv(det_path)
        nonhat = [r for r in det.values() if int(r["label_true"]) == 0 and r.get("prob_pred")]
        falsely_triggered = [r for r in nonhat if float(r["prob_pred"]) >= 0.2]
        print(f"  label=0 images            : {len(nonhat)}")
        print(f"  prob_before >= 0.2 (would trigger mitigation): "
              f"{len(falsely_triggered)}  ({len(falsely_triggered)/max(len(nonhat),1)*100:.1f}%)")
    else:
        print("  [skip] --det_csv not found")

    # ------------------------------------------------------------------
    # Pareto CSV (per-image TR% vs BERTScore)
    # ------------------------------------------------------------------
    pareto_path = Path(args.output_csv).with_suffix(".pareto.csv")
    pareto_rows = []
    for j, r in enumerate(hateful_rows):
        entry = {
            "id":          r["id"],
            "label_true":  r["label_true"],
            "prob_before": r.get("prob_before", ""),
            "prob_after":  r.get("prob_after", ""),
            "tr_pct":      f"{tr_per_image[j]*100:.2f}" if j < len(tr_per_image) else "",
            "hate_location": r.get("hate_location", ""),
            "bertscore_f1": "",
            "clip_score":   "",
            "ssim":         "",
        }
        pareto_rows.append(entry)
    with open(pareto_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pareto_rows)
    print(f"\n  Pareto data saved → {pareto_path}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mitigation evaluation for UnHateMemeDL")

    parser.add_argument("--jsonl",       required=True,  help="Eval JSONL file")
    parser.add_argument("--img_dir",     default=None,   help="Dataset root (for image loading)")
    parser.add_argument("--det_csv",     required=True,  help="Detection predictions CSV (from run_detection_eval.py)")
    parser.add_argument("--out_dir",     default="report/mitigated", help="Dir to save mitigated images + JSONs")
    parser.add_argument("--output_csv",  default="report/mitigation_results.csv")
    parser.add_argument("--vlm_name",       default="Qwen/Qwen3.6-27B")
    parser.add_argument("--diffusion_model_name", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--cache_dir",   default=None)

    # Phase flags
    parser.add_argument("--run_mitigation",  action="store_true", help="Phase 1: run pipeline on hateful images")
    parser.add_argument("--run_judge",       action="store_true", help="Phase 2: re-run VLM on mitigated images")
    parser.add_argument("--compute_metrics", action="store_true", help="Phase 3: compute all metrics")
    parser.add_argument("--all",             action="store_true", help="Run all 3 phases")

    args = parser.parse_args()

    if not any([args.run_mitigation, args.run_judge, args.compute_metrics, args.all]):
        parser.error("Specify at least one phase flag: --run_mitigation, --run_judge, --compute_metrics, or --all")

    if args.all or args.run_mitigation:
        if not args.img_dir:
            parser.error("--img_dir is required for --run_mitigation")
        run_mitigation(args)

    if args.all or args.run_judge:
        if not args.img_dir:
            parser.error("--img_dir is required for --run_judge")
        run_judge(args)

    if args.all or args.compute_metrics:
        compute_metrics(args)


if __name__ == "__main__":
    main()

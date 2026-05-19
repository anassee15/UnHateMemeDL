"""
compute_bootstrap_ci.py — Bootstrap confidence intervals for all evaluation metrics.

Reads the per-image CSVs produced by run_full_eval.py and computes 95% CIs
for detection and mitigation metrics via bootstrap resampling (1000 runs).

Saves results to:
  report/full_eval/bootstrap_ci.csv   — one row per model × metric

Usage
-----
  python report/compute_bootstrap_ci.py
  python report/compute_bootstrap_ci.py --n_boot 5000 --ci 99
"""

import csv
import argparse
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
)

EVAL_ROOT = Path(__file__).parent / "full_eval"

FIELDNAMES = [
    "model", "metric", "mean", "ci_lo", "ci_hi", "ci_width", "n_samples",
]


# ============================================================
# Bootstrap helpers
# ============================================================

def bootstrap_det(y_true, y_prob, y_pred, n_boot=1000, ci=95, seed=42):
    """
    Resample detection predictions n_boot times and recompute each metric.
    Returns a dict: metric_name → {mean, ci_lo, ci_hi}.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p

    metric_fns = {
        "auroc":             lambda t, p, h: roc_auc_score(t, p),
        "macro_f1":          lambda t, p, h: f1_score(t, h, average="macro", zero_division=0),
        "accuracy":          lambda t, p, h: accuracy_score(t, h),
        "precision_hateful": lambda t, p, h: precision_score(t, h, pos_label=1, zero_division=0),
        "recall_hateful":    lambda t, p, h: recall_score(t, h, pos_label=1, zero_division=0),
        "f1_hateful":        lambda t, p, h: f1_score(t, h, pos_label=1, zero_division=0),
    }

    boot = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, fn in metric_fns.items():
            try:
                boot[k].append(fn(y_true[idx], y_prob[idx], y_pred[idx]))
            except Exception:
                pass

    result = {}
    for k, vals in boot.items():
        arr = np.array(vals)
        result[k] = {
            "mean":  float(arr.mean()),
            "ci_lo": float(np.percentile(arr, lo_p)),
            "ci_hi": float(np.percentile(arr, hi_p)),
            "n":     n,
        }
    return result


def bootstrap_col(values, n_boot=1000, ci=95, seed=42):
    """
    Resample a numeric array n_boot times and bootstrap the mean.
    Returns {mean, ci_lo, ci_hi}.
    """
    vals = np.array([v for v in values if not np.isnan(float(v))], dtype=float)
    if len(vals) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}

    rng  = np.random.default_rng(seed)
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    arr  = np.array(boot)
    return {
        "mean":  float(vals.mean()),
        "ci_lo": float(np.percentile(arr, lo_p)),
        "ci_hi": float(np.percentile(arr, hi_p)),
        "n":     len(vals),
    }


# ============================================================
# Load CSV helpers
# ============================================================

def load_det_csv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows
             if r.get("prob_pred") and r.get("label_pred")
             and not r.get("error")]
    y_true = np.array([int(r["label_true"])   for r in valid])
    y_prob = np.array([float(r["prob_pred"])   for r in valid])
    y_pred = np.array([int(r["label_pred"])    for r in valid])
    return y_true, y_prob, y_pred


def load_mit_col(path: Path, col: str):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    hateful = [r for r in rows
               if int(r.get("label_true", 0)) == 1
               and r.get(col) and not r.get("error")]
    try:
        return [float(r[col]) for r in hateful]
    except ValueError:
        return []


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI for eval metrics")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--ci",     type=int, default=95,   help="Confidence interval level (%%)")
    parser.add_argument("--eval_root", default=str(EVAL_ROOT), help="Path to full_eval directory")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out_path  = eval_root.parent / "bootstrap_ci.csv"

    model_dirs = [d for d in eval_root.iterdir() if d.is_dir()]
    if not model_dirs:
        print(f"[error] No model directories found in {eval_root}")
        return

    rows = []
    sep  = "=" * 60

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        det_path   = model_dir / "detection_predictions.csv"
        mit_path   = model_dir / "mitigation_results.csv"

        print(f"\n{sep}")
        print(f"Model: {model_name}")
        print(sep)

        # ----------------------------------------------------------
        # Detection bootstrap
        # ----------------------------------------------------------
        if det_path.exists():
            y_true, y_prob, y_pred = load_det_csv(det_path)
            print(f"  Detection: {len(y_true)} samples — bootstrapping ({args.n_boot}x)…")
            det_ci = bootstrap_det(y_true, y_prob, y_pred,
                                   n_boot=args.n_boot, ci=args.ci)

            print(f"  {'Metric':<22}  {'Mean':>7}  {'CI lo':>7}  {'CI hi':>7}  {'Width':>7}")
            print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
            for metric, d in det_ci.items():
                width = d["ci_hi"] - d["ci_lo"]
                print(f"  {metric:<22}  {d['mean']:>7.4f}  {d['ci_lo']:>7.4f}  {d['ci_hi']:>7.4f}  {width:>7.4f}")
                rows.append({
                    "model":     model_name,
                    "metric":    metric,
                    "mean":      round(d["mean"],  4),
                    "ci_lo":     round(d["ci_lo"], 4),
                    "ci_hi":     round(d["ci_hi"], 4),
                    "ci_width":  round(width,      4),
                    "n_samples": d["n"],
                })
        else:
            print(f"  [skip] No detection CSV at {det_path}")

        # ----------------------------------------------------------
        # Mitigation bootstrap
        # ----------------------------------------------------------
        if mit_path.exists():
            mit_metrics = {
                "prob_after":     load_mit_col(mit_path, "prob_after"),
                "pct_nonhateful": [(1 if float(v) < 0.5 else 0) * 100
                                   for v in load_mit_col(mit_path, "prob_after")],
                "bertscore_f1":   load_mit_col(mit_path, "bertscore_f1"),
                "clip_score":     load_mit_col(mit_path, "clip_score"),
                "ssim":           load_mit_col(mit_path, "ssim"),
                "mps":            load_mit_col(mit_path, "mps"),
            }

            print(f"\n  Mitigation bootstrapping ({args.n_boot}x)…")
            print(f"  {'Metric':<22}  {'Mean':>7}  {'CI lo':>7}  {'CI hi':>7}  {'Width':>7}  {'N':>5}")
            print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}")

            for metric, values in mit_metrics.items():
                if not values:
                    print(f"  {metric:<22}  {'n/a':>7}")
                    continue
                d     = bootstrap_col(values, n_boot=args.n_boot, ci=args.ci)
                width = d["ci_hi"] - d["ci_lo"] if not np.isnan(d["ci_hi"]) else float("nan")
                print(f"  {metric:<22}  {d['mean']:>7.4f}  {d['ci_lo']:>7.4f}  "
                      f"{d['ci_hi']:>7.4f}  {width:>7.4f}  {d['n']:>5}")
                rows.append({
                    "model":     model_name,
                    "metric":    f"mit_{metric}",
                    "mean":      round(d["mean"],  4) if not np.isnan(d["mean"]) else "",
                    "ci_lo":     round(d["ci_lo"], 4) if not np.isnan(d["ci_lo"]) else "",
                    "ci_hi":     round(d["ci_hi"], 4) if not np.isnan(d["ci_hi"]) else "",
                    "ci_width":  round(width,      4) if not np.isnan(width) else "",
                    "n_samples": d["n"],
                })
        else:
            print(f"  [skip] No mitigation CSV at {mit_path}")

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{sep}")
    print(f"Saved {len(rows)} rows → {out_path}")
    print(sep)


if __name__ == "__main__":
    main()

"""
compute_bootstrap_ci.py — Bootstrap confidence intervals for detection & mitigation metrics.

For each model in report/full_eval/, resamples predictions 1000 times (with replacement)
and recomputes every metric to produce 95% confidence intervals.

Outputs
-------
  report/full_eval/bootstrap_ci.csv   one row per (model, metric) with mean, lo, hi, ci_width

Usage
-----
  python src/eval/compute_bootstrap_ci.py
  python src/eval/compute_bootstrap_ci.py --n_boot 2000 --ci 99
  python src/eval/compute_bootstrap_ci.py --eval_root report/full_eval --output report/full_eval/bootstrap_ci.csv
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        precision_score, recall_score,
    )
except ImportError:
    print("[error] scikit-learn not found. Run: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)


# ============================================================
# Bootstrap helpers
# ============================================================

def bootstrap_det(rows: list[dict], n_boot: int, ci: float, seed: int) -> dict:
    """
    Resample detection predictions n_boot times and compute metric CIs.
    Each resample draws len(rows) rows with replacement, then recomputes
    AUROC, Macro-F1, Accuracy, Precision, Recall, F1-hateful.
    Returns {metric: {mean, lo, hi, ci_width}}.
    """
    rng = np.random.default_rng(seed)

    valid = [r for r in rows
             if r.get("prob_pred") and not r.get("error")
             and r.get("label_pred") and r.get("label_true")]
    if not valid:
        return {}

    y_true = np.array([int(r["label_true"])   for r in valid])
    y_prob = np.array([float(r["prob_pred"])   for r in valid])
    y_pred = np.array([int(r["label_pred"])    for r in valid])
    n      = len(valid)

    metric_fns = {
        "auroc":             lambda t, p, h: roc_auc_score(t, p),
        "macro_f1":          lambda t, p, h: f1_score(t, h, average="macro",  zero_division=0),
        "accuracy":          lambda t, p, h: accuracy_score(t, h),
        "precision_hateful": lambda t, p, h: precision_score(t, h, pos_label=1, zero_division=0),
        "recall_hateful":    lambda t, p, h: recall_score(t, h,    pos_label=1, zero_division=0),
        "f1_hateful":        lambda t, p, h: f1_score(t, h,        pos_label=1, zero_division=0),
    }

    boot: dict[str, list[float]] = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, fn in metric_fns.items():
            try:
                boot[k].append(fn(y_true[idx], y_prob[idx], y_pred[idx]))
            except Exception:
                pass

    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p
    return _summarise(boot, lo_p, hi_p)


def bootstrap_col(values: np.ndarray, n_boot: int, ci: float, seed: int) -> dict:
    """
    Bootstrap the mean of a 1-D array.
    Returns {mean, lo, hi, ci_width}.
    """
    rng = np.random.default_rng(seed)
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "ci_width": float("nan")}
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p
    return _summarise({"val": boot}, lo_p, hi_p)["val"]


def _summarise(boot: dict, lo_p: float, hi_p: float) -> dict:
    result = {}
    for k, vals in boot.items():
        arr  = np.array(vals)
        mean = float(arr.mean())
        lo   = float(np.percentile(arr, lo_p))
        hi   = float(np.percentile(arr, hi_p))
        result[k] = {
            "mean":     round(mean, 5),
            "lo":       round(lo, 5),
            "hi":       round(hi, 5),
            "ci_width": round(hi - lo, 5),
        }
    return result


# ============================================================
# CSV loaders
# ============================================================

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_models(eval_root: Path) -> list[tuple[str, Path]]:
    """Return [(model_label, model_dir)] from summary.csv."""
    summary_path = eval_root / "summary.csv"
    if not summary_path.exists():
        print(f"[error] summary.csv not found at {summary_path}", file=sys.stderr)
        sys.exit(1)
    models = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["vlm_name"].split("/")[-1]
            models.append((label, eval_root / row["model_slug"]))
    return models


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for detection & mitigation metrics")
    parser.add_argument("--eval_root", default="report/full_eval", type=Path)
    parser.add_argument("--output",    default=None, type=Path,
                        help="Output CSV path (default: <eval_root>/bootstrap_ci.csv)")
    parser.add_argument("--n_boot",    default=1000, type=int, help="Number of resamples")
    parser.add_argument("--ci",        default=95.0, type=float, help="Confidence interval %%")
    parser.add_argument("--seed",      default=42,   type=int)
    args = parser.parse_args()

    out_path = args.output or (args.eval_root / "bootstrap_ci.csv")
    models   = load_models(args.eval_root)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"Bootstrap CI  —  {args.n_boot} resamples  —  {args.ci}% CI")
    print(sep)

    output_rows = []

    for model_label, model_dir in models:
        print(f"\n  Model: {model_label}")
        print(f"  {'─'*50}")

        # ----------------------------------------------------------
        # Detection metrics
        # ----------------------------------------------------------
        det_rows = load_csv(model_dir / "detection_predictions.csv")
        if det_rows:
            det_ci = bootstrap_det(det_rows, args.n_boot, args.ci, args.seed)
            print(f"\n  Detection  ({len(det_rows)} images):")
            _print_table(det_ci)
            for metric, vals in det_ci.items():
                output_rows.append({
                    "model":    model_label,
                    "phase":    "detection",
                    "metric":   metric,
                    **vals,
                })
        else:
            print("  [skip] No detection CSV found.")

        # ----------------------------------------------------------
        # Mitigation metrics (bootstrapped from per-image columns)
        # ----------------------------------------------------------
        mit_rows = load_csv(model_dir / "mitigation_results.csv")
        hateful  = [r for r in mit_rows
                    if r.get("label_true") == "1"
                    and r.get("prob_after") and not r.get("error")]

        if hateful:
            def _col(key):
                return np.array([float(r[key]) for r in hateful
                                 if r.get(key) and r[key] not in ("", "nan")],
                                dtype=float)

            prob_after = _col("prob_after")
            mit_metrics = {
                "prob_after":     bootstrap_col(prob_after,                           args.n_boot, args.ci, args.seed),
                "pct_nonhateful": bootstrap_col((prob_after < 0.5).astype(float)*100, args.n_boot, args.ci, args.seed),
                "bertscore_f1":   bootstrap_col(_col("bertscore_f1"),                 args.n_boot, args.ci, args.seed),
                "clip_score":     bootstrap_col(_col("clip_score"),                   args.n_boot, args.ci, args.seed),
                "ssim":           bootstrap_col(_col("ssim"),                         args.n_boot, args.ci, args.seed),
                "mps":            bootstrap_col(_col("mps"),                          args.n_boot, args.ci, args.seed),
            }
            print(f"\n  Mitigation  ({len(hateful)} hateful images):")
            _print_table(mit_metrics)
            for metric, vals in mit_metrics.items():
                output_rows.append({
                    "model":  model_label,
                    "phase":  "mitigation",
                    "metric": metric,
                    **vals,
                })
        else:
            print("  [skip] No mitigation CSV found.")

    # ----------------------------------------------------------
    # Save to CSV
    # ----------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "phase", "metric", "mean", "lo", "hi", "ci_width"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n{sep}")
    print(f"Saved {len(output_rows)} rows → {out_path}")
    print(sep)


def _print_table(ci_dict: dict) -> None:
    print(f"  {'Metric':<22} {'Mean':>8}  {'95% CI':>18}  {'Width':>8}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*18}  {'─'*8}")
    for metric, vals in ci_dict.items():
        mean = vals['mean']
        lo   = vals['lo']
        hi   = vals['hi']
        w    = vals['ci_width']
        if any(np.isnan(v) for v in [mean, lo, hi]):
            print(f"  {metric:<22}  {'n/a':>8}")
        else:
            print(f"  {metric:<22}  {mean:>8.4f}  [{lo:>7.4f}, {hi:>7.4f}]  {w:>8.4f}")


if __name__ == "__main__":
    main()

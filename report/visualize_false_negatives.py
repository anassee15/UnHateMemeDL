"""
Visualize false negative memes from a detection CSV:
hateful memes (label_true=1) that were predicted as non-hateful (label_pred=0).

Usage:
    python visualize_false_negatives.py [--csv PATH] [--imgs DIR] [--out PATH] [--cols N]
"""

import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

DEFAULT_CSV  = Path(__file__).parent / "detection_predictions_36.csv"
DEFAULT_IMGS = Path(__file__).parent.parent / "data/eval_data"
DEFAULT_OUT  = Path(__file__).parent / "false_negatives.png"


def load_false_negatives(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["label_true"]) == 1 and int(r["label_pred"]) == 0]


def wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def plot_grid(
    false_negatives: list[dict],
    imgs_dir: Path,
    out_path: Path,
    cols: int = 5,
) -> None:
    n = len(false_negatives)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.2, rows * 4.2),
        gridspec_kw={"hspace": 0.6, "wspace": 0.35},
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"False Negatives — {n} hateful memes misclassified as non-hateful",
        fontsize=14, color="white", fontweight="bold", y=0.995,
    )

    axes_flat = axes.flat if rows > 1 or cols > 1 else [axes]

    for ax, record in zip(axes_flat, false_negatives):
        img_rel = record["img"]          # e.g. "img/84756.png"
        img_path = imgs_dir / img_rel

        ax.set_facecolor("#16213e")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#e94560")
            spine.set_linewidth(1.5)

        # --- image ---
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img, aspect="auto")
            except Exception:
                ax.text(0.5, 0.5, "⚠ load error", ha="center", va="center",
                        color="red", transform=ax.transAxes, fontsize=7)
        else:
            ax.text(0.5, 0.5, "image\nnot found", ha="center", va="center",
                    color="#888", transform=ax.transAxes, fontsize=7)

        # --- title: meme ID + confidence ---
        prob = float(record.get("prob_pred", 0))
        ax.set_title(
            f"#{record['id']}   p={prob:.2f}",
            fontsize=7.5, color="#e0e0e0", pad=3,
        )

        # --- caption below: meme text ---
        meme_text = wrap(record.get("text", ""), width=30)
        ax.set_xlabel(
            meme_text,
            fontsize=6, color="#aaaaaa", labelpad=4,
            loc="center",
        )

    # hide unused axes
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {n} false negatives → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize false negative memes.")
    parser.add_argument("--csv",  default=DEFAULT_CSV,  type=Path, help="Path to predictions CSV")
    parser.add_argument("--imgs", default=DEFAULT_IMGS, type=Path, help="Base directory for images")
    parser.add_argument("--out",  default=DEFAULT_OUT,  type=Path, help="Output PNG path")
    parser.add_argument("--cols", default=5,            type=int,  help="Grid columns (default 5)")
    args = parser.parse_args()

    fn = load_false_negatives(args.csv)
    if not fn:
        print("No false negatives found in", args.csv)
        return

    plot_grid(fn, args.imgs, args.out, cols=args.cols)


if __name__ == "__main__":
    main()

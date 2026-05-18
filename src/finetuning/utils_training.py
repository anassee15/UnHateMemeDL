"""Shared helpers for the finetuning scripts (detection / mitigation / cls_head)."""

import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import TrainerCallback


SFT_CSV_FIELDNAMES = [
    "step", "epoch", "loss", "learning_rate", "grad_norm",
    "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
]


# Substrings whose dotted module name excludes the module from LoRA targeting.
# Covers vision encoder + cross-modal connector (Gemma 4 / Qwen-VL naming) and
# lm_head (tied to embed_tokens; adapting it causes gradient instability).
VISION_KEYWORDS = (
    "vision", "visual", "patch_embed", "image_tower", "img_encoder",
    "siglip", "clip", "vit", "multi_modal_projector",
    "vision_tower", "merger",
    "lm_head",
)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(pool: list, key_fn, val_ratio: float = 0.1, seed: int = 42):
    """Split pool into (train, val) preserving per-key ratios."""
    rng = random.Random(seed)
    buckets: dict = {}
    for rec in pool:
        buckets.setdefault(key_fn(rec), []).append(rec)
    train, val = [], []
    for recs in buckets.values():
        rng.shuffle(recs)
        cut = int(len(recs) * (1 - val_ratio))
        train.extend(recs[:cut])
        val.extend(recs[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compute_warmup_steps(train_size: int, batch_size: int, grad_accum: int,
                         num_epochs: int, ratio: float) -> int:
    steps_per_epoch = math.ceil(train_size / (batch_size * grad_accum))
    return max(1, int(ratio * steps_per_epoch * num_epochs))


def find_lm_linear_names(model) -> list:
    """Full dotted paths of Linear layers in the LM trunk (vision tower excluded)."""
    names = []
    for name, module in model.named_modules():
        if any(kw in name for kw in VISION_KEYWORDS):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if len(name.split(".")[-1]) > 2:
            names.append(name)
    return names


def mask_prefix_and_pad(input_ids: torch.Tensor, prefix_lengths: list,
                        pad_id) -> torch.Tensor:
    """SFT labels: -100 on the prefix (user turn + image patches) and on padding.

    Clamps prefix_len to seq_len-1 so a truncated row still has at least one
    assistant token contributing to the loss (otherwise: NaN).
    """
    labels = input_ids.clone()
    seq_len = input_ids.shape[1]
    for i, prefix_len in enumerate(prefix_lengths):
        labels[i, :min(prefix_len, seq_len - 1)] = -100
    if pad_id is not None:
        labels[input_ids == pad_id] = -100
    return labels


class SFTMetricsLogger(TrainerCallback):
    """Appends train/eval loss to CSV and re-plots training curves on each eval."""

    def __init__(self, output_dir, title: str, fieldnames=SFT_CSV_FIELDNAMES):
        self.output_dir = Path(output_dir)
        self.title = title
        self.fieldnames = fieldnames
        self.csv_path = self.output_dir / "training_metrics.csv"
        self._header_written = self.csv_path.exists()

    def on_log(self, _args, state, _control, logs=None, **_kw):
        if logs is None or not state.is_world_process_zero:
            return
        row = {"step": state.global_step,
               **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames,
                                    extrasaction="ignore", restval="")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def on_evaluate(self, _args, state, _control, **_kw):
        if state.is_world_process_zero:
            self._plot()

    def _plot(self):
        if not self.csv_path.exists():
            return
        steps, train_loss, eval_rows = [], [], []
        with open(self.csv_path) as f:
            for row in csv.DictReader(f):
                s = int(row["step"])
                if row.get("loss"):
                    steps.append(s)
                    train_loss.append(float(row["loss"]))
                if row.get("eval_loss"):
                    eval_rows.append((s, float(row["eval_loss"])))
        fig, ax = plt.subplots(figsize=(8, 4))
        if train_loss:
            ax.plot(steps[:len(train_loss)], train_loss, label="Train loss", alpha=0.7)
        if eval_rows:
            ex, ey = zip(*eval_rows)
            ax.plot(ex, ey, label="Val loss", marker="o", linewidth=2)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss (SFT)")
        ax.set_title(self.title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close(fig)

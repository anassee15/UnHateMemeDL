"""
QLoRA fine-tuning script — Bloc 1: Hateful Meme Detection.

Architecture & design choices
──────────────────────────────
- 4-bit NF4 quantisation (QLoRA): loads a ~27B VLM in ≈14-17 GB VRAM instead of
  ≈54 GB, leaving headroom for activations and image tokens on a single A100 80 GB.
  Ref: Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
       https://arxiv.org/abs/2305.14314

- LoRA applied to the language model only (vision encoder frozen): the visual
  encoder already produces rich multimodal features; the detection bottleneck is
  LLM-side cross-modal reasoning.  This mirrors the design of LLaVA (Liu et al.,
  2023, https://arxiv.org/abs/2304.08485) and InstructBLIP (Dai et al., 2023,
  https://arxiv.org/abs/2305.06500).

- SFT loss (cross-entropy, assistant tokens only):
      L = -(1/|T_a|) Σ_{t∈T_a} log P(t_i | t_{<i}, image)
  Tokens from the user turn and image patches have label=-100 (ignored).
  Ref: Ouyang et al. (2022) "Training language models to follow instructions..."
       https://arxiv.org/abs/2203.02155

- Hateful Memes Challenge dataset: purposely built with "benign confounders"
  (same image / neutral text, same text / neutral image) to prevent unimodal
  shortcuts. Ref: Kiela et al. (2020) https://arxiv.org/abs/2005.04790

Usage:
    python src/finetuning/train_detection.py \
        --model_name google/gemma-4-31b-it \
        --train_jsonl data/hateful-meme/train.jsonl \
        --val_jsonl   data/hateful-meme/dev.jsonl \
        --img_dir     data/hateful-meme \
        --output_dir  checkpoints/detect \
        --cache_dir   <hf-cache-dir> \
        --balance
"""

import sys
import csv
import json
import random
import logging
import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))
from prompt import HATEFUL_DETECTION_PROMPT
from utils import parse_hateful_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_train_val_split(
    train_jsonl: str,
    dev_jsonl: str,
    exclude_jsonl: str | None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Builds train and val lists from the hateful-meme dataset.

    Strategy:
    - Load train.jsonl (all labeled) + dev.jsonl minus any IDs in exclude_jsonl.
    - Shuffle and split 90/10 for internal train/val.
    - exclude_jsonl (eval_490_balanced.jsonl) is never included — it is the
      sacred held-out test set used only for final evaluation.
    """
    exclude_ids: set = set()
    if exclude_jsonl:
        for item in load_jsonl(exclude_jsonl):
            exclude_ids.add(item["id"])

    pool: list[dict] = []

    for item in load_jsonl(train_jsonl):
        if item.get("label") is not None and item["id"] not in exclude_ids:
            pool.append(item)

    for item in load_jsonl(dev_jsonl):
        if item.get("label") is not None and item["id"] not in exclude_ids:
            pool.append(item)

    rng = random.Random(seed)
    rng.shuffle(pool)
    split = int(len(pool) * (1 - val_ratio))
    train_data, val_data = pool[:split], pool[split:]

    label_count = lambda data: {  # noqa: E731
        k: sum(1 for x in data if x["label"] == k) for k in (0, 1)
    }
    logger.info(f"Train: {len(train_data)} examples {label_count(train_data)}")
    logger.info(f"Val  : {len(val_data)} examples {label_count(val_data)}")
    if exclude_ids:
        logger.info(f"Excluded {len(exclude_ids)} eval IDs (eval_490_balanced) from both splits")

    return train_data, val_data


class HatefulMemeDataset(Dataset):
    """
    Wraps a list of hateful-meme records into SFT chat examples.

    Each example is a single-turn conversation:
        user:      [image] + HATEFUL_DETECTION_PROMPT
        assistant: {"classification": "hateful|non-hateful", "probability": 1.0|0.0}

    No system prompt — matches the inference call in vlm.detect_hateful_meme.
    """

    def __init__(self, data: list[dict], img_dir: str, balance: bool = False):
        self.img_dir = Path(img_dir)

        if balance:
            hateful = [x for x in data if x["label"] == 1]
            non_hateful = [x for x in data if x["label"] == 0]
            n = min(len(hateful), len(non_hateful))
            random.shuffle(hateful)
            random.shuffle(non_hateful)
            data = hateful[:n] + non_hateful[:n]
            random.shuffle(data)
            logger.info(f"Balanced: {n} hateful + {n} non-hateful = {len(data)} total")

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_str = "hateful" if item["label"] == 1 else "non-hateful"
        probability = 1.0 if item["label"] == 1 else 0.0
        image = Image.open(self.img_dir / item["img"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": HATEFUL_DETECTION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps({"classification": label_str, "probability": probability}),
            },
        ]
        return {"messages": messages, "image": image, "label": item["label"]}


# ---------------------------------------------------------------------------
# Collator — builds input_ids + labels with proper loss masking
# ---------------------------------------------------------------------------

def collate_fn(batch, processor, max_length: int):
    """
    Applies chat template, tokenises, and masks user/image tokens from the loss.

    The loss is computed only on tokens generated by the assistant role:
        labels[i, :prefix_len] = -100   (user turn + image patches)
        labels[padding positions] = -100

    This implements the SFT objective from Ouyang et al. (2022).
    """
    images = [item["image"] for item in batch]
    messages_list = [item["messages"] for item in batch]

    full_texts, prefix_texts = [], []
    for messages in messages_list:
        full_texts.append(
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
        prefix_texts.append(
            processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
        )

    full_inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Compute exact prefix lengths (including image tokens) per example
    prefix_lengths = []
    for text, image in zip(prefix_texts, images):
        enc = processor(text=[text], images=[image], return_tensors="pt", padding=False)
        prefix_lengths.append(enc["input_ids"].shape[1])

    labels = full_inputs["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    for i, prefix_len in enumerate(prefix_lengths):
        labels[i, :prefix_len] = -100
    if pad_id is not None:
        labels[full_inputs["input_ids"] == pad_id] = -100

    full_inputs["labels"] = labels
    return full_inputs


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

VISION_KEYWORDS = ("vision", "visual", "patch_embed", "image_tower", "img_encoder")


def find_lm_linear_names(model) -> list[str]:
    """
    Returns the names of all Linear layers in the language-model trunk.
    Vision encoder layers are excluded (frozen, not quantised to NF4).

    Design rationale: adapting only the LM trunk is sufficient for cross-modal
    reasoning tasks — see LLaVA-1.5 (Liu et al., 2023) which shows that
    instruction-tuning the LLM while freezing CLIP achieves SOTA on 11 benchmarks.
    """
    names = set()
    for name, module in model.named_modules():
        if any(kw in name for kw in VISION_KEYWORDS):
            continue
        if isinstance(module, torch.nn.Linear) and len(name.split(".")[-1]) > 2:
            names.add(name.split(".")[-1])
    return list(names)


# ---------------------------------------------------------------------------
# Callback: save metrics CSV + plot training curves after each eval
# ---------------------------------------------------------------------------

class MetricsLogger(TrainerCallback):
    """Appends eval metrics to a CSV and re-plots training curves after each eval."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_path = output_dir / "training_metrics.csv"
        self._header_written = self.csv_path.exists()

    def on_log(self, _args, state, _control, logs=None, **_kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        row = {"step": state.global_step,
               **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def on_evaluate(self, _args, state, _control, **_kwargs):
        if state.is_world_process_zero:
            self._plot_curves()

    def _plot_curves(self):
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
        ax.set_title("Detection adapter — QLoRA training curves")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Post-training generative evaluation: F1 + AUROC + plots
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_f1(model, processor, val_dataset, max_new_tokens: int = 64,
                output_dir: Path = None):
    """
    Runs generation on the val set and computes F1 / AUROC.

    Why generative eval and not loss-based?
    The SFT loss measures token-level perplexity, not classification accuracy.
    A model can have low loss but still misclassify if it outputs a well-formatted
    JSON with the wrong label. Generative eval measures what matters at inference.
    Ref: Kiela et al. (2020) use AUROC as the primary metric for this benchmark.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for item in val_dataset:
        messages = item["messages"][:-1]  # user turn only (no assistant)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[item["image"]], return_tensors="pt", padding=False
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        output = processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]

        is_hateful, probability, _ = parse_hateful_response(output)
        y_true.append(item["label"])
        y_pred.append(1 if is_hateful else 0)
        y_prob.append(probability)

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    report = classification_report(y_true, y_pred, target_names=["non-hateful", "hateful"])
    logger.info(f"\n{report}")
    logger.info(f"F1={f1:.4f}  AUROC={auroc:.4f}")

    if output_dir is not None:
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump({"f1": f1, "auroc": auroc, "n_examples": len(y_true)}, f, indent=2)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Probability distribution by ground-truth class
        probs = np.array(y_prob)
        labels_arr = np.array(y_true)
        for label, name, color in [(0, "non-hateful", "steelblue"), (1, "hateful", "tomato")]:
            axes[0].hist(probs[labels_arr == label], bins=20, alpha=0.6,
                         label=name, color=color, density=True)
        axes[0].set_xlabel("Predicted probability")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Confidence distribution by class")
        axes[0].legend()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["non-hateful", "hateful"]).plot(ax=axes[1])
        axes[1].set_title(f"Confusion matrix  (F1={f1:.3f}, AUROC={auroc:.3f})")

        fig.tight_layout()
        fig.savefig(output_dir / "eval_results.png", dpi=150)
        plt.close(fig)
        logger.info(f"Eval plots saved to {output_dir / 'eval_results.png'}")

    return f1, auroc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-4-31b-it")
    parser.add_argument("--train_jsonl", required=True,
                        help="Path to train.jsonl (hateful-meme format).")
    parser.add_argument("--dev_jsonl", required=True,
                        help="Path to dev.jsonl. IDs in --exclude_jsonl are removed.")
    parser.add_argument("--exclude_jsonl", default=None,
                        help="JSONL whose IDs are excluded from train+dev "
                             "(i.e. eval_490_balanced.jsonl). These form the sacred test set.")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of (train+dev minus exclude) held out for val.")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank. r=16 is a good default; r=32 gives more "
                             "capacity at ~2x adapter size. Hu et al. (2022).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Scaling factor. Effective LR ∝ alpha/r. "
                             "Keeping alpha=2*r is a common heuristic.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="QLoRA typically uses 1e-4 to 3e-4.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Effective batch = batch_size * grad_accum.")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap the training set size (e.g. 2000 for fast debug runs). "
                             "None = use all available data.")
    parser.add_argument("--balance", action="store_true",
                        help="Undersample majority class to 50/50.")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip post-training generative F1 eval.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Processor ---
    logger.info(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- 4-bit model (QLoRA) ---
    # NF4 is the information-theoretically optimal 4-bit dtype for normally
    # distributed weights. Double quantisation further reduces the memory
    # footprint of quantisation constants by ~0.37 bits/param.
    # (Dettmers et al., 2023 — QLoRA)
    logger.info("Loading model in 4-bit NF4 (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # --- LoRA ---
    # r=16 introduces ~0.5% trainable parameters. Hu et al. (2022) show that
    # low-rank adapters with r=4-16 match full fine-tuning on most NLP tasks.
    target_modules = find_lm_linear_names(model)
    logger.info(f"LoRA target modules ({len(target_modules)}): {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Datasets ---
    # Use all labeled data (train + dev minus eval IDs), split 90/10 internally.
    train_data, val_data = build_train_val_split(
        train_jsonl=args.train_jsonl,
        dev_jsonl=args.dev_jsonl,
        exclude_jsonl=args.exclude_jsonl,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.max_train_samples and args.max_train_samples < len(train_data):
        train_data = train_data[:args.max_train_samples]
        logger.info(f"Capped training set to {len(train_data)} examples (--max_train_samples)")
    train_ds = HatefulMemeDataset(train_data, args.img_dir, balance=args.balance)
    val_ds = HatefulMemeDataset(val_data, args.img_dir, balance=False)

    # --- Trainer ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,           # keep last 3 checkpoints on disk
        load_best_model_at_end=True,  # restore best checkpoint (min eval_loss)
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,     # 0 avoids PIL + fork issues on the cluster
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, processor=processor, max_length=args.max_length),
        callbacks=[MetricsLogger(output_dir)],
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # --- Save LoRA adapter only (not the full 4-bit base) ---
    adapter_path = output_dir / "adapter_detect"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    logger.info(f"LoRA adapter saved to {adapter_path}")

    # --- Post-training generative eval: F1 + AUROC + plots ---
    if not args.skip_final_eval:
        logger.info("Running post-training generative evaluation (F1 / AUROC)...")
        evaluate_f1(model, processor, val_ds, output_dir=output_dir)


if __name__ == "__main__":
    main()

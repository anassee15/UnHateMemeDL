"""
QLoRA fine-tuning script — Bloc 2: Mitigation prompt generation.

Architecture & design choices
──────────────────────────────
- QLoRA (Dettmers et al., 2023, https://arxiv.org/abs/2305.14314): 4-bit NF4
  base + bfloat16 LoRA adapters. Fits Gemma 4 31B / Qwen3-VL 27B on a single
  A100 80GB or even 40GB at this batch size. `--bf16` switches to full bf16
  LoRA (no quantisation) when fidelity is preferred and VRAM allows.

- LoRA on the language model trunk only (vision encoder frozen): cross-modal
  reasoning happens in the LM, the vision encoder already extracts rich
  features. Mirrors LLaVA (Liu et al., 2023, https://arxiv.org/abs/2304.08485).

- SFT objective (Ouyang et al., 2022, https://arxiv.org/abs/2203.02155):
  cross-entropy on assistant tokens only; user turn + image patches are masked
  to -100 so the model is graded on its mitigation JSON, not on echoing the
  prompt.

- Target JSON is the tight 5-field schema (see prompt.GET_DIFFUSION_PROMPT):
      {hate_source, hate_location, diffusion_prompt, original_text, replacement_text}
  `hate_source` is emitted first so the model's own description acts as a
  chain-of-thought scaffold before the structured fields — known to improve
  structured output quality.

- Training data: data/finetuning/mitigation.jsonl (produced by format_dataset.py).
  Already filtered to hateful-only records with valid diffusion_prompt and
  eval IDs excluded — no post-loading filtering needed here.

Usage:
    python src/finetuning/train_mitigation.py \
        --model_name google/gemma-4-31B-it \
        --dataset_jsonl data/finetuning/mitigation.jsonl \
        --output_dir   checkpoints/mitigation \
        --cache_dir    <hf-cache-dir>
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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))
from prompt import GET_DIFFUSION_PROMPT
from utils import parse_prompt_generation

logger = logging.getLogger(__name__)

# Fixed CSV columns — covers all fields emitted by Trainer's on_log callbacks.
_CSV_FIELDNAMES = [
    "step", "epoch", "loss", "learning_rate", "grad_norm",
    "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
]

HATE_LOCATIONS = ("VISUAL_ONLY", "TEXT_ONLY", "COMBINED", "INTERSECTIONAL")


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_train_val_split(
    dataset_jsonl: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Load mitigation.jsonl (pre-filtered by format_dataset.py) → 90/10 shuffle-split.
    """
    pool = load_jsonl(dataset_jsonl)

    rng = random.Random(seed)
    rng.shuffle(pool)
    split = int(len(pool) * (1 - val_ratio))
    train_data, val_data = pool[:split], pool[split:]

    def loc_dist(data):
        return {loc: sum(1 for x in data if x.get("target", {}).get("hate_location") == loc)
                for loc in HATE_LOCATIONS}

    logger.info(f"Loaded {len(pool)} records from {dataset_jsonl}")
    logger.info(f"Train: {len(train_data)} | hate_location: {loc_dist(train_data)}")
    logger.info(f"Val  : {len(val_data)} | hate_location: {loc_dist(val_data)}")
    return train_data, val_data


def build_target_json(record: dict) -> str:
    """Serialises the pre-built target dict to JSON."""
    return json.dumps(record["target"], ensure_ascii=False)


class MitigationDataset(Dataset):
    """
    Wraps records into single-turn SFT chat examples.

      user:      [image] + 'Meme text: "..."\\n\\n' + GET_DIFFUSION_PROMPT
      assistant: {"hate_source": ..., "hate_location": ..., "diffusion_prompt": ...,
                  "original_text": ..., "replacement_text": ...}

    No system prompt — keeps the inference shape identical to the training
    shape (the FT model is called via vlm.get_diffusion_prompt(use_ft_prompt=True),
    which also passes no system prompt).
    """

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["img_path"]).convert("RGB")
        meme_text = item.get("text", "") or ""
        user_text = f'Meme text: "{meme_text}"\n\n{GET_DIFFUSION_PROMPT}'

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": build_target_json(item),
            },
        ]
        return {
            "messages": messages,
            "image": image,
            "hate_location": item.get("target", {}).get("hate_location"),
        }


# ── Collator ──────────────────────────────────────────────────────────────────

def _is_qwen(processor) -> bool:
    return "Qwen" in processor.__class__.__name__


def collate_fn(batch, processor, max_length: int):
    """
    Apply chat template, tokenize, mask user/image tokens from the loss.

    Qwen-VL processors accept a flat list of images, Gemma 4's processor
    expects List[List[Image]] (one inner list per text example). Branch on
    processor class name.
    """
    messages_list = [item["messages"] for item in batch]
    if _is_qwen(processor):
        images = [item["image"] for item in batch]
    else:
        images = [[item["image"]] for item in batch]

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

    # Exact per-example prefix length (incl. image patches) for loss masking.
    prefix_lengths = []
    for text, image in zip(prefix_texts, images):
        if _is_qwen(processor):
            enc = processor(text=[text], images=[image], return_tensors="pt", padding=False)
        else:
            enc = processor(text=[text], images=image, return_tensors="pt", padding=False)
        prefix_lengths.append(enc["input_ids"].shape[1])

    labels = full_inputs["input_ids"].clone()
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_id = tok.pad_token_id
    seq_len = full_inputs["input_ids"].shape[1]
    for i, prefix_len in enumerate(prefix_lengths):
        # Guard: if the full sequence was truncated to max_length, the
        # untruncated prefix_len can exceed seq_len, masking all assistant
        # tokens and producing NaN loss. Clamp to seq_len - 1 so at least
        # one assistant token is always eligible for the loss.
        labels[i, :min(prefix_len, seq_len - 1)] = -100
    if pad_id is not None:
        labels[full_inputs["input_ids"] == pad_id] = -100

    full_inputs["labels"] = labels
    return full_inputs


# ── LoRA target selection ─────────────────────────────────────────────────────

# Substrings whose dotted module name causes the module to be excluded from LoRA.
# Covers: vision encoder + cross-modal connector (Gemma 4 / Qwen-VL naming) +
# lm_head, which is tied to embed_tokens (tie_word_embeddings=True); adapting a
# tied output layer with LoRA causes gradient instability and PEFT warnings.
VISION_KEYWORDS = (
    "vision", "visual", "patch_embed", "image_tower", "img_encoder",
    "siglip", "clip", "vit", "multi_modal_projector",
    "vision_tower", "merger",
    "lm_head",
)


def find_lm_linear_names(model) -> list[str]:
    """
    Full dotted paths of Linear layers in the LM trunk (vision tower excluded).
    Uses full paths (not leaf names) so PEFT cannot accidentally match same-named
    vision modules.
    """
    names = []
    for name, module in model.named_modules():
        if any(kw in name for kw in VISION_KEYWORDS):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if len(name.split(".")[-1]) > 2:
            names.append(name)
    return names


# ── Metrics callback (mirrors train_detection.py) ─────────────────────────────

class MetricsLogger(TrainerCallback):
    """Appends train/eval metrics to a CSV and re-plots curves after each eval."""

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
            writer = csv.DictWriter(
                f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore", restval=""
            )
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
        ax.set_title("Mitigation adapter — LoRA training curves")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close(fig)


# ── Post-training generative eval ─────────────────────────────────────────────

@torch.inference_mode()
def evaluate_mitigation(model, processor, val_dataset, max_new_tokens: int = 320,
                        output_dir: Path = None):
    """
    Generate on the val set, parse JSON, and report:
      - JSON validity rate (parse succeeds, no fallback)
      - schema completeness (all 5 fields present and well-typed)
      - hate_location accuracy + confusion matrix (4-class)

    Why not perplexity? The SFT loss is token-level. A well-formatted JSON with
    the wrong hate_location can still have low loss. What matters at inference
    is whether `mitigate_image` can act on the parsed plan — which requires
    valid JSON + sensible hate_location + non-empty diffusion_prompt.
    """
    model.eval()
    n = len(val_dataset)
    n_valid = 0
    n_complete = 0
    y_true_loc, y_pred_loc = [], []
    bad_examples = []

    for idx in range(n):
        item = val_dataset[idx]
        messages = item["messages"][:-1]  # user only
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if _is_qwen(processor):
            inputs = processor(text=[text], images=[item["image"]], return_tensors="pt", padding=False)
        else:
            inputs = processor(text=[text], images=[[item["image"]]], return_tensors="pt", padding=False)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        output = processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]

        parsed = parse_prompt_generation(output)
        is_valid = "_parse_error" not in parsed
        if is_valid:
            n_valid += 1

        required_str = ["hate_source", "hate_location", "diffusion_prompt"]
        complete = all(
            isinstance(parsed.get(k), str) and parsed.get(k).strip()
            for k in required_str
        ) and parsed.get("hate_location") in HATE_LOCATIONS
        if complete:
            n_complete += 1

        gt_loc = item["hate_location"]
        pred_loc = parsed.get("hate_location")
        if gt_loc in HATE_LOCATIONS and pred_loc in HATE_LOCATIONS:
            y_true_loc.append(gt_loc)
            y_pred_loc.append(pred_loc)

        if not complete and len(bad_examples) < 10:
            bad_examples.append({"idx": idx, "raw": output[:400], "parsed": parsed})

    validity_rate = n_valid / max(n, 1)
    completeness_rate = n_complete / max(n, 1)
    loc_accuracy = accuracy_score(y_true_loc, y_pred_loc) if y_true_loc else float("nan")

    logger.info(f"JSON validity:   {validity_rate:.3f}  ({n_valid}/{n})")
    logger.info(f"Schema complete: {completeness_rate:.3f}  ({n_complete}/{n})")
    logger.info(f"hate_location accuracy: {loc_accuracy:.3f}  (on {len(y_true_loc)} valid preds)")

    if output_dir is not None:
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump({
                "n_examples": n,
                "json_validity_rate": validity_rate,
                "schema_completeness_rate": completeness_rate,
                "hate_location_accuracy": loc_accuracy,
                "n_loc_evaluable": len(y_true_loc),
                "bad_examples": bad_examples,
            }, f, indent=2)

        if y_true_loc:
            cm = confusion_matrix(y_true_loc, y_pred_loc, labels=list(HATE_LOCATIONS))
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=list(HATE_LOCATIONS)).plot(
                ax=ax, xticks_rotation=30
            )
            ax.set_title(f"hate_location — acc={loc_accuracy:.3f}")
            fig.tight_layout()
            fig.savefig(output_dir / "eval_hate_location_cm.png", dpi=150)
            plt.close(fig)

    return validity_rate, completeness_rate, loc_accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-4-31B-it",
                        help="HF model id. Tested with Gemma 4 31B-it and Qwen3-VL family.")
    parser.add_argument("--dataset_jsonl", default="data/finetuning/mitigation.jsonl",
                        help="Path to data/finetuning/mitigation.jsonl produced by format_dataset.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true",
                        help="Use full bfloat16 LoRA instead of QLoRA NF4. Higher VRAM, "
                             "no quantisation noise.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max token length.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap the training set (e.g. 50 for smoke runs).")
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Processor ─────────────────────────────────────────────────────────────
    logger.info(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Model: QLoRA by default; --bf16 swaps to full bf16 LoRA ───────────────
    if args.bf16:
        quant_config = None
        logger.info("Loading model in bfloat16 (no quantisation, LoRA only)...")
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading model in 4-bit NF4 (QLoRA, bf16 compute)...")

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    if quant_config is not None:
        # Prepares the int4 base for stable gradient flow through LoRA
        # (casts norms to fp32, enables input grads, etc.)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.enable_input_require_grads()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    target_modules = find_lm_linear_names(model)
    logger.info(f"LoRA target modules ({len(target_modules)} matched)")

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

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_data, val_data = build_train_val_split(
        dataset_jsonl=args.dataset_jsonl,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.max_train_samples and args.max_train_samples < len(train_data):
        train_data = train_data[:args.max_train_samples]
        logger.info(f"Capped train set to {len(train_data)} (--max_train_samples)")
    train_ds = MitigationDataset(train_data)
    val_ds = MitigationDataset(val_data)

    # ── Trainer ───────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
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

    # ── Save adapter only ─────────────────────────────────────────────────────
    adapter_path = output_dir / "adapter_mitigation"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    logger.info(f"LoRA adapter saved to {adapter_path}")

    # ── Post-training generative eval ─────────────────────────────────────────
    if not args.skip_final_eval:
        logger.info("Running post-training generative evaluation...")
        evaluate_mitigation(model, processor, val_ds, output_dir=output_dir)


if __name__ == "__main__":
    main()

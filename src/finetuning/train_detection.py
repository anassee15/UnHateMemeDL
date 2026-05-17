"""
LoRA fine-tuning script — Bloc 1: Hateful Meme Detection.

Architecture & design choices

- Full-precision LoRA (bfloat16): loads the model weights in bfloat16, keeping
  all computation in native precision. Requires more VRAM than QLoRA but avoids
  quantisation noise and is faster per step.
  Ref: Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
       https://arxiv.org/abs/2106.09685

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

- Rich SFT targets from detection_full.jsonl (produced by format_dataset.py):
    hateful:     {classification, description, hate_type, hate_location}
    non-hateful: {classification, description}
  Including description (chain-of-thought) and hate_type/hate_location gives the
  model more training signal per example and teaches structured reasoning.

- Meme text is included explicitly in the user turn ("Meme text: ...") to remove
  the OCR bottleneck, mirroring the approach in train_mitigation.py.

- QLoRA (default) vs full bf16 LoRA (--bf16): pass --bf16 to skip quantisation
  for higher fidelity when VRAM allows; default is 4-bit NF4 QLoRA which fits
  Gemma 4 31B on a single A100 40GB.

- Catastrophic forgetting prevention: VISION_KEYWORDS explicitly covers the
  vision encoder top-level module (vision_tower) and the cross-modal connector
  (merger, multi_modal_projector) so LoRA is never applied to those modules.
  Adapting the connector is the primary cause of visual-capability collapse
  observed in prior runs.

Usage:
    # QLoRA (default, lower VRAM)
    python src/finetuning/train_detection.py \
        --model_name google/gemma-4-31b-it \
        --dataset_jsonl data/finetuning/detection_full.jsonl \
        --output_dir  checkpoints/detect \
        --cache_dir   <hf-cache-dir> \
        --balance

    # Full bf16 LoRA (higher fidelity, more VRAM)
    python src/finetuning/train_detection.py ... --bf16
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
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))
from prompt import HATEFUL_DETECTION_PROMPT_FT_RICH
from utils import parse_hateful_response

logger = logging.getLogger(__name__)

_CSV_FIELDNAMES = [
    "step", "epoch", "loss", "learning_rate", "grad_norm",
    "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
]


#  Dataset 

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_train_val_split(dataset_jsonl: str, val_ratio: float = 0.1, seed: int = 42):
    """
    Load detection_full.jsonl (pre-filtered by format_dataset.py) → 90/10 shuffle-split.
    """
    pool = load_jsonl(dataset_jsonl)

    rng = random.Random(seed)
    rng.shuffle(pool)
    split = int(len(pool) * (1 - val_ratio))
    train_data, val_data = pool[:split], pool[split:]

    def label_count(data):
        return {k: sum(1 for x in data if x.get("label") == k) for k in (0, 1)}

    logger.info(f"Loaded {len(pool)} records from {dataset_jsonl}")
    logger.info(f"Train: {len(train_data)} {label_count(train_data)}")
    logger.info(f"Val  : {len(val_data)} {label_count(val_data)}")
    return train_data, val_data


def _build_assistant_response(item: dict) -> str:
    """
    Build the rich SFT target JSON from a detection_full.jsonl record.

    Hateful:     {classification, description, hate_type, hate_location}
    Non-hateful: {classification, description}
    """
    target = item["target"]
    label = item.get("label")

    assistant = {
        "classification": target["classification"],
        "description": target.get("description", ""),
    }
    if label == 1:
        if target.get("hate_type"):
            assistant["hate_type"] = target["hate_type"]
        if target.get("hate_location"):
            assistant["hate_location"] = target["hate_location"]

    return json.dumps(assistant, ensure_ascii=False)


class HatefulMemeDataset(Dataset):
    """
    Wraps detection_full.jsonl records into SFT chat examples.

    Each example is a single-turn conversation:
        user:      [image] + 'Meme text: "..."' + HATEFUL_DETECTION_PROMPT_FT_RICH
        assistant: {classification, description[, hate_type, hate_location]}

    No system prompt — matches the inference call in vlm.detect_hateful_meme.
    """

    def __init__(self, data: list, balance: bool = False):
        if balance:
            hateful = [x for x in data if x.get("label") == 1]
            non_hateful = [x for x in data if x.get("label") == 0]
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
        image = Image.open(item["img_path"]).convert("RGB")
        meme_text = item.get("text", "") or ""
        user_text = f'Meme text: "{meme_text}"\n\n{HATEFUL_DETECTION_PROMPT_FT_RICH}'

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
                "content": _build_assistant_response(item),
            },
        ]
        return {"messages": messages, "image": image, "label": item.get("label")}


#  Collator 

def collate_fn(batch, processor, max_length: int):
    """
    Applies chat template, tokenises, and masks user/image tokens from the loss.

    The loss is computed only on tokens generated by the assistant role:
        labels[i, :prefix_len] = -100   (user turn + image patches)
        labels[padding positions] = -100

    This implements the SFT objective from Ouyang et al. (2022).
    """
    images = [[item["image"]] for item in batch]
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

    prefix_lengths = []
    for text, image in zip(prefix_texts, images):
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


#  LoRA helpers

VISION_KEYWORDS = (
    "vision", "visual", "patch_embed", "image_tower", "img_encoder",
    "siglip", "clip", "vit", "multi_modal_projector",
    "vision_tower",  # Gemma 4 / Qwen3-VL top-level vision encoder
    "merger",        # Qwen3-VL cross-modal connector; adapting it collapses visual grounding
    "lm_head",       # tied to embed_tokens; LoRA on tied layers causes gradient instability
)


def find_lm_linear_names(model) -> list:
    """
    Returns full dotted paths of all Linear layers in the language-model trunk.
    Vision encoder layers are excluded (frozen).

    Uses full paths (not leaf names) so that PEFT targets exactly these modules
    and cannot accidentally match same-named layers in the vision encoder
    (e.g. Gemma4ClippableLinear wrappers that share leaf names like q_proj).

    Design rationale: adapting only the LM trunk is sufficient for cross-modal
    reasoning tasks — see LLaVA-1.5 (Liu et al., 2023) which shows that
    instruction-tuning the LLM while freezing CLIP achieves SOTA on 11 benchmarks.
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


#  Metrics callback 

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
        ax.set_title("Detection adapter — LoRA training curves")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close(fig)


#  Post-training generative eval 

@torch.inference_mode()
def evaluate_f1(model, processor, val_dataset, max_new_tokens: int = 128,
                output_dir: Path = None):
    """
    Runs generation on the val set and computes F1 / AUROC.

    Why generative eval and not loss-based?
    The SFT loss measures token-level perplexity, not classification accuracy.
    A model can have low loss but still misclassify if it outputs a well-formatted
    JSON with the wrong label. Generative eval measures what matters at inference.
    Ref: Kiela et al. (2020) use AUROC as the primary metric for this benchmark.

    max_new_tokens is bumped to 128 (vs 64 previously) to accommodate the richer
    JSON response that now includes description + optional hate_type/hate_location.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for item in val_dataset:
        messages = item["messages"][:-1]  # user turn only
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[[item["image"]]], return_tensors="pt", padding=False
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        output = processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]

        try:
            is_hateful, probability, _ = parse_hateful_response(output)
        except (ValueError, TypeError):
            is_hateful, probability = False, 0.0

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

        probs = np.array(y_prob)
        labels_arr = np.array(y_true)
        for label, name, color in [(0, "non-hateful", "steelblue"), (1, "hateful", "tomato")]:
            axes[0].hist(probs[labels_arr == label], bins=20, alpha=0.6,
                         label=name, color=color, density=True)
        axes[0].set_xlabel("Predicted probability")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Confidence distribution by class")
        axes[0].legend()

        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["non-hateful", "hateful"]).plot(ax=axes[1])
        axes[1].set_title(f"Confusion matrix  (F1={f1:.3f}, AUROC={auroc:.3f})")

        fig.tight_layout()
        fig.savefig(output_dir / "eval_results.png", dpi=150)
        plt.close(fig)
        logger.info(f"Eval plots saved to {output_dir / 'eval_results.png'}")

    return f1, auroc


#  Main 

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-4-31B-it")
    parser.add_argument("--dataset_jsonl", default="data/finetuning/detection_full.jsonl",
                        help="Path to detection_full.jsonl produced by format_dataset.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of dataset held out for validation.")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank. r=16 is a good default; r=32 gives more "
                             "capacity at ~2x adapter size. Hu et al. (2022).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Scaling factor. Effective LR ∝ alpha/r. "
                             "Keeping alpha=2*r is a common heuristic.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="LoRA typically uses 1e-4 to 2e-4.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Effective batch = batch_size * grad_accum.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Bumped from 768 to accommodate the longer rich target JSON.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap the training set size (e.g. 200 for smoke runs).")
    parser.add_argument("--bf16", action="store_true",
                        help="Use full bfloat16 LoRA instead of QLoRA NF4. "
                             "Higher VRAM, no quantisation noise. "
                             "Default: QLoRA 4-bit NF4 (fits 31B on A100 40GB).")
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

    #  Processor 
    logger.info(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Model: QLoRA by default; --bf16 switches to full bfloat16 LoRA ───────────
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
    model.config.use_cache = False
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()

    #  LoRA 
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

    #  Datasets 
    train_data, val_data = build_train_val_split(
        dataset_jsonl=args.dataset_jsonl,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.max_train_samples and args.max_train_samples < len(train_data):
        train_data = train_data[:args.max_train_samples]
        logger.info(f"Capped training set to {len(train_data)} examples (--max_train_samples)")
    train_ds = HatefulMemeDataset(train_data, balance=args.balance)
    val_ds = HatefulMemeDataset(val_data, balance=False)

    #  Trainer 
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

    #  Save adapter 
    adapter_path = output_dir / "adapter_detect"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    logger.info(f"LoRA adapter saved to {adapter_path}")

    #  Post-training generative eval 
    if not args.skip_final_eval:
        logger.info("Running post-training generative evaluation (F1 / AUROC)...")
        evaluate_f1(model, processor, val_ds, output_dir=output_dir)


if __name__ == "__main__":
    main()

"""
Classification head fine-tuning script — Bloc 1: Hateful Meme Detection.

Approach
─────────

The entire VLM (vision encoder + multimodal projector + LM trunk) is kept
completely frozen. A single Linear(hidden_dim → 1) head is trained on top of
the last-token hidden state that the frozen model produces when given:

    [image] + HATEFUL_DETECTION_PROMPT_FT  (user turn only, no generation)

Loss: binary cross-entropy (BCEWithLogitsLoss).
Output: sigmoid(logit) ∈ [0, 1] used directly as the hateful probability.

Why this is safer than LoRA for a 31B model
────────────────────────────────────────────

- Zero risk of catastrophic forgetting: base weights never change.
- Only ~hidden_dim parameters are trainable (≈7 K for a 7168-dim model).
- Memory: 31 B x 2 bytes ≈ 62 GB (bfloat16) + negligible head — fits A100 80 GB.
- No activation graph is stored for the frozen VLM (torch.no_grad inside forward),
  so per-step memory is close to pure inference.
- Larger effective batch size than LoRA because no VLM gradients to accumulate.

Usage:
    python src/finetuning/train_cls_head.py \
        --model_name google/gemma-4-31b-it \
        --train_jsonl data/hateful-meme/train.jsonl \
        --dev_jsonl   data/hateful-meme/dev.jsonl \
        --img_dir     data/hateful-meme \
        --output_dir  checkpoints/cls_head \
        --cache_dir   <hf-cache-dir> \
        --balance
"""

import sys
import csv
import json
import math
import random
import logging
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import (
    f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))
from prompt import HATEFUL_DETECTION_PROMPT_FT

logger = logging.getLogger(__name__)

_CSV_FIELDNAMES = [
    "step", "epoch", "loss", "learning_rate", "grad_norm",
    "eval_loss", "eval_f1", "eval_auroc",
    "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
]


#  Data utilities 

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
    Merges train.jsonl + dev.jsonl, removes eval IDs, splits 90/10.
    The exclude_jsonl (eval_490_balanced.jsonl) is the sacred held-out test set.
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

    # Stratify by binary label so val class balance is preserved exactly.
    # BestHeadSaver selects checkpoints on eval_f1, which is sensitive to
    # class-ratio drift in small (~10%) val splits.
    rng = random.Random(seed)
    by_label: dict = {}
    for item in pool:
        by_label.setdefault(item["label"], []).append(item)
    train_data, val_data = [], []
    for recs in by_label.values():
        rng.shuffle(recs)
        split = int(len(recs) * (1 - val_ratio))
        train_data.extend(recs[:split])
        val_data.extend(recs[split:])
    rng.shuffle(train_data)
    rng.shuffle(val_data)

    label_count = lambda data: {k: sum(1 for x in data if x["label"] == k) for k in (0, 1)}  # noqa: E731
    logger.info(f"Train: {len(train_data)} examples {label_count(train_data)}")
    logger.info(f"Val  : {len(val_data)} examples {label_count(val_data)}")
    if exclude_ids:
        logger.info(f"Excluded {len(exclude_ids)} eval IDs from both splits")

    return train_data, val_data


#  Dataset 

class HatefulMemeDataset(Dataset):
    """
    Returns raw (image, binary_label) pairs.

    Simpler than the SFT dataset: no chat template, no assistant response.
    The chat template is applied in the collator so the processor can handle
    image token insertion correctly alongside the text.
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
        image = Image.open(self.img_dir / item["img"]).convert("RGB")
        return {"image": image, "label": item["label"]}


#  Collator 

def collate_fn(batch, processor, max_length: int):
    """
    Builds model inputs from (image, label) pairs.

    Only the user turn is included — no assistant response.
    The model does a forward pass and the head reads the last token hidden state,
    so generation is never involved.
    """
    # Gemma 4 processor expects List[List[Image]] — one inner list per text example.
    # A flat List[Image] is interpreted as one example with N images, not N examples.
    images = [[item["image"]] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)

    # One identical prompt per example — only the image differs
    user_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": HATEFUL_DETECTION_PROMPT_FT},
                ],
            }
        ]
        for _ in batch
    ]
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in user_messages
    ]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs["labels"] = labels
    return inputs


#  Model 

class VLMWithClassificationHead(nn.Module):
    """
    Frozen VLM + trainable binary classification head.

    Step-by-step forward pass:
      1. Run the frozen VLM with output_hidden_states=True (inside torch.no_grad
         so no activation graph is stored — equivalent to inference memory cost).
      2. Extract the hidden state of the last non-padding token from the final
         transformer layer. For a decoder-only model this token has attended to
         the full sequence (image patches + prompt text) and carries the most
         information for a classification decision.
      3. Pass through Linear(hidden_dim → 1) → scalar logit.
      4. Compute BCEWithLogitsLoss against the binary label if provided.
    """

    def __init__(self, vlm: nn.Module, hidden_dim: int):
        super().__init__()
        self.vlm = vlm

        # Small head: one linear layer, bfloat16 to match VLM activations.
        # Both weight and bias are zero-initialized so that the initial logit is
        # exactly 0 → sigmoid(0) = 0.5 → BCE loss = log(2) ≈ 0.693, regardless
        # of hidden state magnitude. trunc_normal would produce large initial
        # logits (std ≈ 5 for hidden_dim=7168) and a loss of 5-8, meaning the
        # head starts confidently wrong and wastes early training steps recovering.
        self.classifier = nn.Linear(hidden_dim, 1, dtype=torch.bfloat16)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        # Step 1 — frozen VLM forward (no gradient graph, saves ~62 GB of
        # intermediate activations that would otherwise be retained for backprop)
        with torch.no_grad():
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )

        # Step 2 — last non-padding token of the final transformer layer
        last_hidden = outputs.hidden_states[-1]          # (B, T, D)
        if attention_mask is not None:
            # attention_mask is 1 for real tokens, 0 for padding (right-padded)
            seq_lengths = attention_mask.sum(dim=1) - 1  # index of last real token
        else:
            seq_lengths = torch.full(
                (last_hidden.size(0),), last_hidden.size(1) - 1,
                device=last_hidden.device, dtype=torch.long,
            )
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lengths]     # (B, D)

        # Step 3 — classification head
        logits = self.classifier(pooled)                 # (B, 1)

        # Step 4 — loss
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels.to(logits.dtype)
            )

        return SequenceClassifierOutput(loss=loss, logits=logits)


#  Callback: save best head only

class BestHeadSaver(TrainerCallback):
    """
    Saves only the classifier head (28 KB) whenever val F1 improves.

    Why not use Trainer's built-in save_strategy?
    Trainer would try to save the full frozen VLM (~62 GB) at each checkpoint,
    which hits two problems: prohibitive disk usage and a safetensors error on
    tied weights (embed_tokens / lm_head share the same tensor in Gemma 4).
    Since the VLM is frozen, checkpointing it is pointless — only the head changes.
    """

    def __init__(self, model: "VLMWithClassificationHead", output_dir: Path):
        self.model = model
        self.best_path = output_dir / "best_classifier.pt"
        self.best_f1 = -1.0

    def on_evaluate(self, _args, state, _control, metrics=None, **_kwargs):
        if not state.is_world_process_zero or metrics is None:
            return
        f1 = metrics.get("eval_f1", -1.0)
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(self.model.classifier.state_dict(), self.best_path)
            logger.info(f"New best F1={f1:.4f} — head saved to {self.best_path}")


#  Callback: CSV logging + training curves

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
        train_steps, train_loss, eval_rows_loss, eval_rows_f1 = [], [], [], []
        with open(self.csv_path) as f:
            for row in csv.DictReader(f):
                s = int(row["step"])
                if row.get("loss"):
                    train_steps.append(s)
                    train_loss.append(float(row["loss"]))
                if row.get("eval_loss"):
                    eval_rows_loss.append((s, float(row["eval_loss"])))
                if row.get("eval_f1"):
                    eval_rows_f1.append((s, float(row["eval_f1"])))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        if train_loss:
            ax.plot(train_steps, train_loss, label="Train loss", alpha=0.7)
        if eval_rows_loss:
            ex, ey = zip(*eval_rows_loss)
            ax.plot(ex, ey, label="Val loss", marker="o", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("BCE loss")
        ax.set_title("Classification head — loss")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        if eval_rows_f1:
            ex, ey = zip(*eval_rows_f1)
            ax.plot(ex, ey, label="Val F1", marker="o", color="tab:green", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("F1")
        ax.set_title("Classification head — val F1")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close(fig)


#  Metrics for Trainer 

def compute_metrics(eval_pred):
    """
    Called by Trainer after each eval step.
    eval_pred.predictions: logits (N, 1) as numpy array
    eval_pred.label_ids:   binary labels (N,) as numpy array
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).squeeze(-1).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    return {"f1": f1, "auroc": auroc}


#  Post-training evaluation 

@torch.inference_mode()
def evaluate_final(model, val_dataset, collate, output_dir: Path):
    """Full classification report + confusion matrix + probability histogram."""
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate, shuffle=False)

    model.eval()
    y_true, y_prob = [], []

    for batch in loader:
        batch = {k: v.to(next(model.parameters()).device) if hasattr(v, "to") else v
                 for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            out = model(**batch)
        probs = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.cpu().numpy().tolist())

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    y_true = [int(l) for l in y_true]

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    report = classification_report(y_true, y_pred, target_names=["non-hateful", "hateful"])
    logger.info(f"\n{report}")
    logger.info(f"F1={f1:.4f}  AUROC={auroc:.4f}")

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump({"f1": f1, "auroc": auroc, "n_examples": len(y_true)}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    probs_arr = np.array(y_prob)
    labels_arr = np.array(y_true)
    for label, name, color in [(0, "non-hateful", "steelblue"), (1, "hateful", "tomato")]:
        axes[0].hist(probs_arr[labels_arr == label], bins=20, alpha=0.6,
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
    parser.add_argument("--model_name", default="google/gemma-4-31b-it")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--exclude_jsonl", default=None,
                        help="IDs to exclude (eval_490_balanced.jsonl — the sacred test set).")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Head-only training uses a higher LR than LoRA (1e-3 to 1e-4).")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="A linear head on frozen features converges in 1-3 epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Larger than LoRA is possible — no VLM gradients stored.")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Effective batch = batch_size * grad_accum.")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Must fit image tokens + prompt. Gemma 4 uses up to ~1024 image tokens.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--balance", action="store_true",
                        help="Undersample majority class to 50/50.")
    parser.add_argument("--skip_final_eval", action="store_true")
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

    #  Frozen VLM 
    # Loaded in bfloat16. All parameters are immediately frozen — only the
    # classification head (added below) will be trained.
    logger.info("Loading VLM in bfloat16 (fully frozen)...")
    vlm = AutoModelForMultimodalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    vlm.config.use_cache = False

    for param in vlm.parameters():
        param.requires_grad = False
    logger.info("All VLM parameters frozen.")

    #  Hidden dimension 
    # Multimodal models store the LM hidden size under text_config; fall back
    # to the top-level hidden_size for single-config models.
    cfg = vlm.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden_dim = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")
    logger.info(f"LM hidden dimension: {hidden_dim}")

    #  Classification head 
    # Wrap the frozen VLM with a trainable Linear(hidden_dim → 1) head.
    model = VLMWithClassificationHead(vlm, hidden_dim)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.4f}%)")

    #  Datasets 
    train_data, val_data = build_train_val_split(
        train_jsonl=args.train_jsonl,
        dev_jsonl=args.dev_jsonl,
        exclude_jsonl=args.exclude_jsonl,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.max_train_samples and args.max_train_samples < len(train_data):
        train_data = train_data[:args.max_train_samples]
        logger.info(f"Capped training set to {len(train_data)} examples")

    train_ds = HatefulMemeDataset(train_data, args.img_dir, balance=args.balance)
    val_ds = HatefulMemeDataset(val_data, args.img_dir, balance=False)

    collate = partial(collate_fn, processor=processor, max_length=args.max_length)

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_train_steps = steps_per_epoch * args.num_epochs
    warmup_steps = max(1, int(0.05 * total_train_steps))

    #  Trainer
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=True,
        # Never save full model checkpoints: the frozen VLM is 62 GB and has
        # tied weights (embed_tokens / lm_head) that safetensors rejects.
        # BestHeadSaver below saves only the 28 KB classifier on each F1 improvement.
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    best_head_saver = BestHeadSaver(model, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLogger(output_dir), best_head_saver],
    )

    logger.info("Starting classification head training...")
    trainer.train()

    #  Save artefacts
    # best_classifier.pt — best val-F1 head (written by BestHeadSaver during training)
    # classifier_final.pt — head weights at the last training step
    # processor/          — tokenizer + image processor for inference
    head_path = output_dir / "cls_head"
    head_path.mkdir(exist_ok=True)
    torch.save(model.classifier.state_dict(), head_path / "classifier_final.pt")
    processor.save_pretrained(str(head_path))
    logger.info(
        f"Best head  (F1={best_head_saver.best_f1:.4f}): {output_dir / 'best_classifier.pt'}"
    )
    logger.info(f"Final head : {head_path / 'classifier_final.pt'}")

    #  Post-training evaluation
    if not args.skip_final_eval:
        logger.info("Running post-training evaluation on best head (F1 / AUROC)...")
        best_state = torch.load(output_dir / "best_classifier.pt", weights_only=True)
        model.classifier.load_state_dict(best_state)
        evaluate_final(model, val_ds, collate, output_dir=output_dir)


if __name__ == "__main__":
    main()

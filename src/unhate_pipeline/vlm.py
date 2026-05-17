import sys
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

from prompt import (
    HATEFUL_DETECTION_PROMPT, HATEFUL_DETECTION_PROMPT_FT,
    TYPE_OF_HATE_PROMPT, SOURCE_OF_HATE_PROMPT,
    build_diffusion_prompt,
)


def instantiate_vlm(
    model_name: str,
    cache_dir: str | None = None,
    adapter_path: str | None = None,
    mitigation_adapter_path: str | None = None,
) -> tuple:
    """
    Load a VLM in bfloat16, optionally with one or two fine-tuned LoRA adapters.

    adapter_path: detection LoRA adapter (from train_detection.py). Loaded under
                  the name 'detect' if both adapters are provided.
    mitigation_adapter_path: mitigation LoRA adapter (from train_mitigation.py).
                  Loaded under the name 'mitigate'. When provided, callers should
                  activate it before generating diffusion prompts via
                  `model.set_adapter('mitigate')` (handled by get_diffusion_prompt
                  when `use_ft_prompt=True`).

    If only one adapter is given, it is loaded as the single active adapter and
    no name switching is required.
    """
    print("[info] Loading processor...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[info] Loading model in {dtype}...", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    both = adapter_path is not None and mitigation_adapter_path is not None
    if adapter_path is not None or mitigation_adapter_path is not None:
        from peft import PeftModel

    if both:
        print(f"[info] Loading detection adapter from {adapter_path}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name="detect")
        print(f"[info] Loading mitigation adapter from {mitigation_adapter_path}...", file=sys.stderr)
        model.load_adapter(mitigation_adapter_path, adapter_name="mitigate")
        # Default-active: detection. Switched on demand by get_diffusion_prompt.
        model.set_adapter("detect")
    elif adapter_path is not None:
        print(f"[info] Loading detection adapter from {adapter_path}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)
    elif mitigation_adapter_path is not None:
        print(f"[info] Loading mitigation adapter from {mitigation_adapter_path}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, mitigation_adapter_path)

    model.eval()
    return model, processor


# ── Classification head helpers ───────────────────────────────────────────────

def load_cls_head(model: nn.Module, head_path: str) -> nn.Linear:
    """
    Reconstruct the Linear(hidden_dim → 1) head trained by train_cls_head.py
    and load its weights from `head_path` (path to classifier.pt).

    The head is moved to the same device as the VLM and set to eval mode.
    """
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden_dim = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")

    head = nn.Linear(hidden_dim, 1, dtype=torch.bfloat16)
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    head.load_state_dict(state)

    device = next(model.parameters()).device
    head = head.to(device).eval()
    print(f"[info] Classification head loaded (hidden_dim={hidden_dim}) on {device}",
          file=sys.stderr)
    return head


@torch.inference_mode()
def detect_hateful_meme_cls_head(
    model: nn.Module,
    processor,
    head: nn.Linear,
    image_path: str,
) -> tuple[bool, float]:
    """
    Detect whether a meme is hateful using the trained classification head.

    This is a pure forward pass — no token generation.
    Returns (is_hateful: bool, probability: float in [0, 1]).

    Step-by-step:
      1. Build user-only prompt (same template used during training).
      2. Forward through frozen VLM with output_hidden_states=True.
      3. Extract the last non-padding token hidden state (final layer).
      4. Pass through head → sigmoid → probability.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": HATEFUL_DETECTION_PROMPT_FT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[[image]], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]          # (1, T, D)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        seq_len = int(attention_mask.sum(dim=1).item()) - 1
    else:
        seq_len = last_hidden.size(1) - 1

    pooled = last_hidden[0, seq_len]                 # (D,)
    logit = head(pooled.unsqueeze(0))                # (1, 1)
    probability = torch.sigmoid(logit).item()
    is_hateful = probability >= 0.5

    return is_hateful, probability


@torch.inference_mode()
def run_vlm(model, processor, image_path, prompt, thinking=False, max_new_tokens=512, temperature=0.2, system_prompt=""):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})


    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        })

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[[image]], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
    )

    prompt_len = inputs["input_ids"].shape[1]
    output = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
    return output


def detect_hateful_meme(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, HATEFUL_DETECTION_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_modality(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, SOURCE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_type(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, TYPE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def get_diffusion_prompt(
    model, processor, image_path,
    thinking: bool = True, max_new_tokens: int = 512, temperature: float = 0.0,
):
    """
    Generate the mitigation plan JSON using GET_DIFFUSION_PROMPT.

    thinking: when True, includes the step-by-step CoT reasoning block in the
    prompt. Activates the 'mitigate' PEFT adapter if it was loaded alongside
    another adapter.
    """
    if hasattr(model, "peft_config") and "mitigate" in getattr(model, "peft_config", {}):
        model.set_adapter("mitigate")
    prompt = build_diffusion_prompt(thinking=thinking)
    return run_vlm(
        model, processor, image_path, prompt,
        thinking=False, max_new_tokens=max_new_tokens, temperature=temperature,
        system_prompt="",
    )
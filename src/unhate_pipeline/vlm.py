import sys
import re
import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

from prompt import (
    HATEFUL_DETECTION_PROMPT, HATEFUL_DETECTION_PROMPT_FT,
    HATEFUL_DETECTION_PROMPT_FT_RICH,
    TYPE_OF_HATE_PROMPT, SOURCE_OF_HATE_PROMPT,
    build_diffusion_prompt,
    # Detection prompt-ablation pipelines (run_detection_eval.py --pipeline):
    FEWSHOT_SYNTHETIC_PROMPT, BASELINE_V2_PROMPT, SINGLE_AFFECT_PROMPT,
)
# NOTE: import `prompt` before `affect_prompting` — see circular-import note in prompt.py.
from affect_prompting import (
    AFFECT_CLASSIFICATION_PROMPT,
    MEME_CATEGORY_PROMPT,
    build_affect_aware_hateful_detection_prompt,
    build_category_aware_hateful_detection_prompt,
    build_historical_fewshot_detection_prompt,
    build_categorized_v2_detection_prompt,
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


# classification head helpers

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


def _parse_json_safe(raw: str) -> dict:
    """Strip markdown fences and parse JSON; return empty dict on failure."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _parse_affect(raw: str) -> dict:
    """Parse affect classification JSON; return safe defaults on failure."""
    parsed = _parse_json_safe(raw)
    defaults = {
        "overall_sentiment": "neutral",
        "humor": "not_funny",
        "sarcasm": "not_sarcastic",
        "offense": "not_offensive",
        "motivation": "not_motivational",
        "rationale": "",
    }
    defaults.update({k: v for k, v in parsed.items() if k in defaults})
    return defaults


def _parse_category(raw: str) -> str:
    """Parse category classification JSON; return 'general_culture' on failure."""
    parsed = _parse_json_safe(raw)
    category = parsed.get("category", "general_culture")
    if category not in ("historical", "general_culture", "identity_social"):
        return "general_culture"
    return category


def detect_hateful_meme(model, processor, image_path, pipeline=None,
                        thinking=False, max_new_tokens=512, temperature=0.95):
    """
    Classify a meme as hateful or not.

    pipeline=None (default) — main's original behavior: adapter-aware prompt
        selection (HATEFUL_DETECTION_PROMPT, or the FT-rich prompt when a
        detection LoRA adapter is active). Used by main.py / run_full_eval.py.

    Prompt-ablation pipelines (driven by run_detection_eval.py --pipeline):
      "baseline"       — 1 call, 17 synthetic calibration few-shot examples
      "baseline_v2"    — 1 call, compact 4-example real few-shot prompt
      "single_affect"  — 1 call, single prompt with internal affect reasoning
      "affect"         — 2 calls: affect classification → affect-specific detection
      "categorized"    — 2-3 calls: category → (affect) → hate detection
      "categorized_v2" — 2 calls: category → category-specific few-shot detection
    """
    if pipeline == "baseline":
        return run_vlm(model, processor, image_path, FEWSHOT_SYNTHETIC_PROMPT,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "baseline_v2":
        return run_vlm(model, processor, image_path, BASELINE_V2_PROMPT,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "single_affect":
        return run_vlm(model, processor, image_path, SINGLE_AFFECT_PROMPT,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "categorized_v2":
        print("[pipeline:categorized_v2] Step 1 — category classification", file=sys.stderr)
        cat_raw = run_vlm(model, processor, image_path, MEME_CATEGORY_PROMPT,
                          thinking, max_new_tokens, temperature)
        category = _parse_category(cat_raw)
        print(f"[pipeline:categorized_v2] Category: {category}", file=sys.stderr)

        print("[pipeline:categorized_v2] Step 2 — hate classification", file=sys.stderr)
        hate_prompt = build_categorized_v2_detection_prompt(category)
        return run_vlm(model, processor, image_path, hate_prompt,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "affect":
        print("[pipeline:affect] Step 1 — affect classification", file=sys.stderr)
        affect_raw = run_vlm(model, processor, image_path, AFFECT_CLASSIFICATION_PROMPT,
                             thinking, max_new_tokens, temperature)
        affect = _parse_affect(affect_raw)
        print(f"[pipeline:affect] Detected: sentiment={affect['overall_sentiment']} "
              f"humor={affect['humor']} sarcasm={affect['sarcasm']} "
              f"offense={affect['offense']}", file=sys.stderr)

        print("[pipeline:affect] Step 2 — hate classification", file=sys.stderr)
        hate_prompt = build_affect_aware_hateful_detection_prompt(affect)
        return run_vlm(model, processor, image_path, hate_prompt,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "categorized":
        print("[pipeline:categorized] Step 1 — category classification", file=sys.stderr)
        cat_raw = run_vlm(model, processor, image_path, MEME_CATEGORY_PROMPT,
                          thinking, max_new_tokens, temperature)
        category = _parse_category(cat_raw)
        print(f"[pipeline:categorized] Category: {category}", file=sys.stderr)

        if category == "historical":
            # historical memes use few-shot examples directly and skip the affect step
            print("[pipeline:categorized] Step 2 — historical few-shot hate classification", file=sys.stderr)
            hate_prompt = build_historical_fewshot_detection_prompt()
        else:
            print(f"[pipeline:categorized] Step 2 — affect classification for '{category}'", file=sys.stderr)
            affect_raw = run_vlm(model, processor, image_path, AFFECT_CLASSIFICATION_PROMPT,
                                 thinking, max_new_tokens, temperature)
            affect = _parse_affect(affect_raw)
            print(f"[pipeline:categorized] Detected: sentiment={affect['overall_sentiment']} "
                  f"humor={affect['humor']} sarcasm={affect['sarcasm']} "
                  f"offense={affect['offense']}", file=sys.stderr)
            hate_prompt = build_category_aware_hateful_detection_prompt(category, affect)

        print("[pipeline:categorized] Final step — hate classification", file=sys.stderr)
        return run_vlm(model, processor, image_path, hate_prompt,
                       thinking, max_new_tokens, temperature)

    # default path: use the FT-rich prompt when a detection adapter is active, else the base prompt
    is_finetuned = hasattr(model, "peft_config") and len(getattr(model, "peft_config", {})) > 0
    # get_diffusion_prompt may have switched to the mitigate adapter; switch back
    if is_finetuned and "detect" in getattr(model, "peft_config", {}):
        model.set_adapter("detect")
    prompt = HATEFUL_DETECTION_PROMPT_FT_RICH if is_finetuned else HATEFUL_DETECTION_PROMPT
    return run_vlm(model, processor, image_path, prompt, thinking, max_new_tokens, temperature)


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
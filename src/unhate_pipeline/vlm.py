import sys
import re
import json
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from prompt import (
    HATEFUL_DETECTION_PROMPT,
    SINGLE_AFFECT_PROMPT,
    TYPE_OF_HATE_PROMPT,
    SOURCE_OF_HATE_PROMPT,
    GET_DIFFUSION_SYSTEM_PROMPT,
    GET_DIFFUSION_USER_PROMPT,
    BASELINE_V2_PROMPT,
)
from affect_prompting import (
    AFFECT_CLASSIFICATION_PROMPT,
    MEME_CATEGORY_PROMPT,
    build_affect_aware_hateful_detection_prompt,
    build_category_aware_hateful_detection_prompt,
    build_historical_fewshot_detection_prompt,
    build_categorized_v2_detection_prompt,
)


def instantiate_vlm(model_name: str, cache_dir: str | None = None) -> AutoModelForImageTextToText:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[info] Using dtype: {dtype}", file=sys.stderr)
    print("[info] Loading processor...", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    print("[info] Loading model...", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    return model, processor


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
    inputs = processor(text=[text], images=[image], return_tensors="pt")
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


# ---------------------------------------------------------------------------
# Private helpers for multi-step pipeline JSON parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_hateful_meme(model, processor, image_path, pipeline="baseline",
                        thinking=False, max_new_tokens=512, temperature=0.95):
    """
    Classify a meme as hateful or not.

    pipeline options:
      "baseline"       — 1 call, baseline prompt + 17 few-shot examples
      "baseline_v2"    — 1 call, compact 4-example few-shot prompt (Qwen3.6-friendly)
      "single_affect"  — 1 call, single prompt with internal affect reasoning
      "affect"         — 2 calls: affect classification → affect-specific hate detection
      "categorized"    — 2-3 calls: category → (affect) → hate detection
      "categorized_v2" — 2 calls: category → category-specific few-shot hate detection
    """
    if pipeline == "baseline_v2":
        return run_vlm(model, processor, image_path, BASELINE_V2_PROMPT,
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

    elif pipeline == "single_affect":
        return run_vlm(model, processor, image_path, SINGLE_AFFECT_PROMPT,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "affect":
        # Step 1: classify affect/sentiment
        print("[pipeline:affect] Step 1 — affect classification", file=sys.stderr)
        affect_raw = run_vlm(model, processor, image_path, AFFECT_CLASSIFICATION_PROMPT,
                             thinking, max_new_tokens, temperature)
        affect = _parse_affect(affect_raw)
        print(f"[pipeline:affect] Detected: sentiment={affect['overall_sentiment']} "
              f"humor={affect['humor']} sarcasm={affect['sarcasm']} "
              f"offense={affect['offense']}", file=sys.stderr)

        # Step 2: build affect-specific hate detection prompt and classify
        print("[pipeline:affect] Step 2 — hate classification", file=sys.stderr)
        hate_prompt = build_affect_aware_hateful_detection_prompt(affect)
        return run_vlm(model, processor, image_path, hate_prompt,
                       thinking, max_new_tokens, temperature)

    elif pipeline == "categorized":
        # Step 1: classify meme category
        print("[pipeline:categorized] Step 1 — category classification", file=sys.stderr)
        cat_raw = run_vlm(model, processor, image_path, MEME_CATEGORY_PROMPT,
                          thinking, max_new_tokens, temperature)
        category = _parse_category(cat_raw)
        print(f"[pipeline:categorized] Category: {category}", file=sys.stderr)

        if category == "historical":
            # Step 2 (historical): use few-shot examples directly, skip affect
            print("[pipeline:categorized] Step 2 — historical few-shot hate classification", file=sys.stderr)
            hate_prompt = build_historical_fewshot_detection_prompt()
        else:
            # Step 2 (general_culture / identity_social): classify affect first
            print(f"[pipeline:categorized] Step 2 — affect classification for '{category}'", file=sys.stderr)
            affect_raw = run_vlm(model, processor, image_path, AFFECT_CLASSIFICATION_PROMPT,
                                 thinking, max_new_tokens, temperature)
            affect = _parse_affect(affect_raw)
            print(f"[pipeline:categorized] Detected: sentiment={affect['overall_sentiment']} "
                  f"humor={affect['humor']} sarcasm={affect['sarcasm']} "
                  f"offense={affect['offense']}", file=sys.stderr)
            # Step 3: build category+affect-specific prompt
            hate_prompt = build_category_aware_hateful_detection_prompt(category, affect)

        # Final step: hate classification
        print("[pipeline:categorized] Final step — hate classification", file=sys.stderr)
        return run_vlm(model, processor, image_path, hate_prompt,
                       thinking, max_new_tokens, temperature)

    else:  # "baseline" or any unknown value
        return run_vlm(model, processor, image_path, HATEFUL_DETECTION_PROMPT,
                       thinking, max_new_tokens, temperature)


def detect_hate_modality(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, SOURCE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_type(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, TYPE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def get_diffusion_prompt(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, GET_DIFFUSION_USER_PROMPT, thinking, max_new_tokens, temperature, system_prompt=GET_DIFFUSION_SYSTEM_PROMPT)

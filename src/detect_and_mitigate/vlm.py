import sys
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from prompt import HATEFUL_DETECTION_PROMPT, TYPE_OF_HATE_PROMPT, SOURCE_OF_HATE_PROMPT, GET_DIFFUSION_SYSTEM_PROMPT, GET_DIFFUSION_USER_PROMPT


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


def detect_hateful_meme(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, HATEFUL_DETECTION_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_modality(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, SOURCE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_type(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, TYPE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def get_diffusion_prompt(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_vlm(model, processor, image_path, GET_DIFFUSION_USER_PROMPT, thinking, max_new_tokens, temperature, system_prompt=GET_DIFFUSION_SYSTEM_PROMPT)
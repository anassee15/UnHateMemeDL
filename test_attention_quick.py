#!/usr/bin/env python3
"""Quick smoke test: load Gemma-4 with eager attn and check output_attentions=True works."""
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "google/gemma-4-31B-it"
CACHE  = "/scratch/hf_cache"
IMAGE  = "/home/gmikou/project_final/data/eval_data/img/01268.png"
PROMPT = "Is this meme hateful? Answer yes or no."

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"[test] dtype={dtype}", file=sys.stderr)

print("[test] Loading processor...", file=sys.stderr)
processor = AutoProcessor.from_pretrained(MODEL, cache_dir=CACHE, trust_remote_code=True)

print("[test] Loading model (eager)...", file=sys.stderr)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    cache_dir=CACHE,
    trust_remote_code=True,
    dtype=dtype,
    device_map="auto",
    attn_implementation="eager",
)
print("[test] Model loaded.", file=sys.stderr)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image = Image.open(IMAGE).convert("RGB")
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

print("[test] Running forward pass with output_attentions=True ...", file=sys.stderr)
with torch.inference_mode():
    out = model(**inputs, output_attentions=True)

captured = len(out.attentions) if out.attentions is not None else 0
print(f"[test] attention tensors captured = {captured}", file=sys.stderr)

if captured > 0:
    shapes = [tuple(a.shape) for a in out.attentions[:3]]
    print(f"[test] first 3 shapes: {shapes}", file=sys.stderr)
    print("[test] PASS — attention capture works correctly.", file=sys.stderr)
    sys.exit(0)
else:
    print("[test] FAIL — no attention tensors returned even with eager mode.", file=sys.stderr)
    sys.exit(1)

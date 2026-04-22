import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


PROMPT_TEXT = (
    "Hatefulness definition: Hate speech is defined as a direct or indirect attack on people based"
    "on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex,"
    "gender identity, sexual orientation, disability, disease, mocking hate crime or evoking pain from"
    "historical figures. The definition of attack includes violent or dehumanizing speech, statements of"
    "inferiority, and calls for exclusion or segregation.\n\n"
    "Classification criteria: A meme is considered hateful if either the image or text is hateful on its own."
    "The meme is also hateful if both the image and text are not hateful individually,"
    "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and"
    "they do not together create a hateful meaning, the meme is not considered hateful.\n\n"
    "You consider visual and textual meanings to classify the meme as hateful or non-hateful based on"
    "the hatefulness definition and classification criteria. After giving a **short explanation**, provide the"
    "answer with either hateful or non-hateful in the following format:\n"
    "Classification:\n"
    "Probability of the meme being hateful (from 0 to 1):"
)

def build_messages(prompt_text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=False, default="data/")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--disable_thinking", action="store_true")
    args = parser.parse_args()

    print(f"[info] Starting inference with model: {args.model_name}", file=sys.stderr)
    print(f"[info] Data path: {args.data_path}", file=sys.stderr)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[info] Using dtype: {dtype}", file=sys.stderr)
    print("[info] Loading processor...", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    print("[info] Loading model...", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    print(f"[info] Model loaded on device: {model.device}", file=sys.stderr)
    image_paths = sorted(Path(args.data_path).glob("*.png"))
    if not image_paths:
        print(f"[error] No .png images found in: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Found {len(image_paths)} image(s).", file=sys.stderr)
    messages = build_messages(PROMPT_TEXT)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.disable_thinking,
    )

    results = []
    for image_path in image_paths:
        print(f"[info] Running generation for: {image_path.name}", file=sys.stderr)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
        )

        prompt_len = inputs["input_ids"].shape[1]
        output = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
        results.append({"image": str(image_path), "response": output})

    print("[info] Generation complete.", file=sys.stderr)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
from PIL import Image
from utils import HATEFUL_DETECTION_PROMPT, TYPE_OF_HATE_PROMPT, SOURCE_OF_HATE_PROMPT, TEXT_MITIGATION_PROMPT


def run_inference_image(model, processor, image_path, prompt, thinking=False, max_new_tokens=512, temperature=0.2):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
    return run_inference_image(model, processor, image_path, HATEFUL_DETECTION_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_modality(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_inference_image(model, processor, image_path, SOURCE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def detect_hate_type(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_inference_image(model, processor, image_path, TYPE_OF_HATE_PROMPT, thinking, max_new_tokens, temperature)


def mitigate_hateful_text(model, processor, image_path, thinking=False, max_new_tokens=512, temperature=0.95):
    return run_inference_image(model, processor, image_path, TEXT_MITIGATION_PROMPT, thinking, max_new_tokens, temperature)
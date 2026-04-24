import sys
import torch

from diffusers import Flux2KleinPipeline

from utils import REMOVE_TEXT_PROMPT, ADD_TEXT_PROMPT, REPLACE_TEXT_PROMPT, UNHATE_IMAGE_PROMPT


def instantiate_diffusion(model_name: str, cache_dir: str | None =None):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[info] Using dtype: {torch_dtype}", file=sys.stderr)
    print(f"[info] Loading diffusion model from repo: {model_name}", file=sys.stderr)

    pipe = Flux2KleinPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def run_diffusion(pipe, image, prompt):
    image = pipe(
        prompt=prompt,
        image=image,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    ).images[0]
    return image


def remove_text_from_image(pipe, image, text_to_remove=""):
    return run_diffusion(pipe, image, REMOVE_TEXT_PROMPT.format(text_to_remove=text_to_remove))

def add_text_to_image(pipe, image, new_text=""):
    return run_diffusion(pipe, image, ADD_TEXT_PROMPT.format(new_text=new_text))

def replace_text_in_image(pipe, image, old_text="", new_text=""):
    return run_diffusion(pipe, image, REPLACE_TEXT_PROMPT.format(old_text=old_text, new_text=new_text))

def unhate_image(pipe, image, details_of_hate=""):
    return run_diffusion(pipe, image, UNHATE_IMAGE_PROMPT.format(details_of_hate=details_of_hate))
"""
Inference time benchmark for UnHateMemeDL.

Measures per-step latency for the full pipeline on a small random subset:
  1. VLM detection     (detect_hateful_meme)
  2. VLM prompt gen    (get_diffusion_prompt)
  3. Diffusion         (mitigate_image)
  4. Total

A warmup phase runs first and is discarded to avoid cold-start bias
(CUDA kernel compilation, KV-cache warmup, etc.).

Usage
-----
  python src/eval/benchmark_inference.py \\
      --jsonl    data/eval_data/eval_490_balanced.jsonl \\
      --img_dir  <dataset-root> \\
      --n_warmup  3 \\
      --n_measure 15
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "unhate_pipeline"))

from vlm import instantiate_vlm, detect_hateful_meme, get_diffusion_prompt
from diffusion import instantiate_diffusion, mitigate_image
from utils import parse_hateful_response, parse_prompt_generation


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_once(vlm, processor, diffusion_model, generator, image, img_path):
    """Run full pipeline on one image.

    Returns (t_detect, t_prompt, t_diffusion, t_total) in seconds.
    Always runs all steps regardless of prob score so timings are consistent.
    """
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    raw = detect_hateful_meme(vlm, processor, img_path)
    parse_hateful_response(raw)
    t_detect = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_prompt = get_diffusion_prompt(vlm, processor, img_path)
    mitigation = parse_prompt_generation(raw_prompt)
    t_prompt = time.perf_counter() - t1

    t2 = time.perf_counter()
    mitigate_image(diffusion_model, image, mitigation, generator=generator)
    t_diffusion = time.perf_counter() - t2

    t_total = time.perf_counter() - t_start
    return t_detect, t_prompt, t_diffusion, t_total


def main():
    parser = argparse.ArgumentParser(description="Inference time benchmark for UnHateMemeDL")
    parser.add_argument("--jsonl",     required=True, help="Eval JSONL file")
    parser.add_argument("--img_dir",   required=True, help="Dataset root")
    parser.add_argument("--n_warmup",  type=int, default=3,  help="Warmup runs (discarded)")
    parser.add_argument("--n_measure", type=int, default=15, help="Timed runs")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--vlm_name",            default="Qwen/Qwen3.6-27B")
    parser.add_argument("--diffusion_model_name", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    img_root = Path(args.img_dir)

    samples = load_jsonl(Path(args.jsonl))
    # Prefer hateful images — mitigation is only meaningful on them
    hateful = [s for s in samples if int(s["label"]) == 1]
    pool = hateful if len(hateful) >= args.n_warmup + args.n_measure else samples

    n_needed = args.n_warmup + args.n_measure
    if len(pool) < n_needed:
        print(f"[warn] Only {len(pool)} images available, need {n_needed} — some will repeat.",
              file=sys.stderr)
        pool = (pool * ((n_needed // len(pool)) + 1))[:n_needed]
    else:
        pool = random.sample(pool, n_needed)

    warmup_samples  = pool[:args.n_warmup]
    measure_samples = pool[args.n_warmup:]

    sep = "=" * 60
    print(sep)
    print("INFERENCE TIME BENCHMARK — UnHateMemeDL")
    print(sep)
    print(f"  VLM             : {args.vlm_name}")
    print(f"  Diffusion model : {args.diffusion_model_name}")
    print(f"  Device          : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  N warmup        : {args.n_warmup}")
    print(f"  N measured      : {args.n_measure}")
    print(f"  Seed            : {args.seed}")
    print(sep)

    print("\nLoading models...", file=sys.stderr)
    vlm, processor     = instantiate_vlm(args.vlm_name, args.cache_dir)
    diffusion_model    = instantiate_diffusion(args.diffusion_model_name, cache_dir=args.cache_dir)
    device             = "cuda" if torch.cuda.is_available() else "cpu"
    generator          = torch.Generator(device=device).manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    print(f"\nWarmup ({args.n_warmup} runs, results discarded)...")
    for i, sample in enumerate(warmup_samples):
        img_path = img_root / sample["img"]
        image    = Image.open(img_path).convert("RGB")
        run_once(vlm, processor, diffusion_model, generator, image, img_path)
        print(f"  warmup {i+1}/{args.n_warmup} done")

    # ------------------------------------------------------------------
    # Timed runs
    # ------------------------------------------------------------------
    print(f"\nTimed runs ({args.n_measure} runs)...")
    timings: dict[str, list[float]] = {"detect": [], "prompt": [], "diffusion": [], "total": []}

    skipped = 0
    for i, sample in enumerate(measure_samples):
        img_path = img_root / sample["img"]
        image    = Image.open(img_path).convert("RGB")
        try:
            t_detect, t_prompt, t_diffusion, t_total = run_once(
                vlm, processor, diffusion_model, generator, image, img_path
            )
        except Exception as e:
            print(f"  [{i+1:>{len(str(args.n_measure))}}/{args.n_measure}]  SKIPPED ({e})")
            skipped += 1
            continue
        timings["detect"].append(t_detect)
        timings["prompt"].append(t_prompt)
        timings["diffusion"].append(t_diffusion)
        timings["total"].append(t_total)
        print(f"  [{i+1:>{len(str(args.n_measure))}}/{args.n_measure}]  "
              f"total={t_total:.2f}s  "
              f"(detect={t_detect:.2f}s  prompt={t_prompt:.2f}s  diffusion={t_diffusion:.2f}s)")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if not timings["total"]:
        print("\n[error] No successful timed runs — all images were skipped.")
        sys.exit(1)

    print(f"\n{sep}")
    print("RESULTS")
    print(sep)
    if skipped:
        print(f"  Skipped (parse error) : {skipped}/{args.n_measure}")
    print(f"  {'Step':<14}  {'Mean (s)':>9}  {'Std (s)':>9}  {'Min (s)':>9}  {'Max (s)':>9}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    steps = [
        ("Detection",  timings["detect"]),
        ("Prompt gen", timings["prompt"]),
        ("Diffusion",  timings["diffusion"]),
        ("Total",      timings["total"]),
    ]
    for name, vals in steps:
        arr = np.array(vals)
        print(f"  {name:<14}  {arr.mean():>9.2f}  {arr.std():>9.2f}  {arr.min():>9.2f}  {arr.max():>9.2f}")

    print(sep)
    total_arr = np.array(timings["total"])
    print(f"\n  Mean total inference time : {total_arr.mean():.2f}s ± {total_arr.std():.2f}s")
    print(sep)


if __name__ == "__main__":
    main()

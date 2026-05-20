# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UnHateMemeDL mitigates hateful content in image memes using open-source Vision-Language Models (VLMs) and Diffusion Models. Built as a course project for EPFL EE-559 Deep Learning. Runs on EPFL's RCP cluster (run:ai + GPU pods) via a custom Docker image.

## Common commands

Inference pipeline (single run on a directory of `.png` files):

```bash
python3 src/unhate_pipeline/main.py \
  --vlm_name <hf-model-id> \
  --diffusion_model_name <hf-model-id> \
  --data_path data/ \
  --cache_dir <hf-cache-dir>
```

Detection eval (Phase 1 = GPU inference, Phase 2 = CPU metrics, resumable):

```bash
# Full
python src/eval/run_detection_eval.py --jsonl data/eval_data/eval_490_balanced.jsonl --img_dir <dataset-root> --output report/detection_predictions.csv
# Metrics only
python src/eval/run_detection_eval.py --jsonl ... --output ... --metrics_only
# Per-modality F1 add-on
python src/eval/run_detection_eval.py ... --modality_analysis
```

Mitigation eval (3 phases — `--run_mitigation`, `--run_judge`, `--compute_metrics`, or `--all`). Needs extra deps not in `requirements.txt`: `pip install bert_score detoxify scikit-image`.

Docker (build + push to EPFL registry, then submit a `runai` job — see README.md for the full `runai submit` invocation and SSH jumphost details):

```bash
cd docker && docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v<VERSION>
docker push registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v<VERSION>
```

There are no tests, linters, or build steps configured.

## Architecture

Two-stage VLM → Diffusion pipeline orchestrated by `src/unhate_pipeline/main.py`:

1. **Detect** (`vlm.detect_hateful_meme`) — VLM returns a JSON object `{classification, probability, description}` parsed by `utils.parse_hateful_response`. If `probability < 0.2`, the image is passed through unchanged.
2. **Plan mitigation** (`vlm.get_diffusion_prompt`) — VLM, given a system + user prompt from `prompt.py`, returns a JSON mitigation plan with fields `hate_location` (`VISUAL_ONLY` / `TEXT_ONLY` / `COMBINED` / `INTERSECTIONAL` / `STRUCTURAL`), `flux_prompt`, `original_text`, `replacement_text`, etc. Parsed by `utils.parse_prompt_generation` (very lenient — strips `<think>` blocks and markdown fences, fixes Python-style `None/True/False`, falls back to `ast.literal_eval`, and on failure returns a passthrough plan).
3. **Mitigate** (`diffusion.mitigate_image`) — branches on `hate_location`:
   - Visual branch → `run_diffusion` (Flux2KleinPipeline, img2img, 6 steps, sequential CPU offload).
   - Text branch → `handle_text_mitigation`: `erase_text` (diffusion with `ERASE_TEXT_PROMPT`), then `draw_meme_text` to render `replacement_text` at top/bottom positions inferred from the original layout.

Key invariants:
- The VLM is loaded with `AutoModelForImageTextToText` (`device_map="auto"`, `bfloat16` on CUDA). The diffusion pipeline uses `Flux2KleinPipeline` with `enable_sequential_cpu_offload()` — both models share the GPU, so memory is tight.
- All prompts live in `src/unhate_pipeline/prompt.py`. Changing the JSON schema returned by the VLM means updating both the prompt and the corresponding `utils.parse_*` function in lockstep.
- Eval scripts import pipeline modules via `sys.path.insert(0, .../unhate_pipeline)` rather than as a package — preserve that import style if adding new eval scripts.
- Detection eval is **resumable**: it keys on the `id` field in the output CSV and skips already-processed rows. Always append, never overwrite.

## Data layout

- `data/eval_data/*.jsonl` — eval splits (records have `id`, `img`, `label`, `text`; `img` is resolved relative to `--img_dir`).
- `example_mitigated/` — committed before/after image examples.
- `report/` — eval CSVs and metrics summaries (gitignored except for committed examples).
- `src/finetuning/` — placeholder, no code yet.

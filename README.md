# Beyond Detection: Multimodal Mitigation of Hateful Memes

Mitigate hateful content on image meme using open source Vision-Language Models (VLMs) and Diffusion Models.

## Data & models used

**Dataset**: 
- [Facebook Hateful Memes](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset/data) 

**Models**: 
- [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Gemma-4b](https://huggingface.co/google/gemma-4-31B-it)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)


## Structure of the repository

Two-stage pipeline: a **VLM** detects hateful memes and plans a mitigation, then a **diffusion model** rewrites the image.

```
report/                 # poster and report pdfs, screencast of pipeline inference
data/                   # datasets (source memes, eval split, fine-tuning data)
docker/                 # image used to run on the EPFL RCP cluster
models/                 # training configs for every fine-tuned model we tested
src/
  unhate_pipeline/      # core two-stage pipeline (VLM -> diffusion)
  eval/                 # evaluation scripts (detection / mitigation / grid)
  finetuning/           # LoRA / classification-head fine-tuning
```

### `src/unhate_pipeline/` — the pipeline

| File | Role |
|---|---|
| `main.py` | Entry point. Two-phase run on a folder of `.png`: phase 1 loads the VLM and, for every image, runs detection + mitigation plan; the VLM is then released; phase 2 loads the diffusion model on the full GPU and produces the mitigated images. |
| `vlm.py` | Loads the VLM (optionally with LoRA adapters or a classification head), runs hateful detection (incl. the prompt-ablation pipelines) and generates the diffusion mitigation plan. |
| `diffusion.py` | Loads the FLUX diffusion pipeline and applies the mitigation: visual edit (img2img) and/or text removal + redraw. |
| `prompt.py` | All VLM prompt templates: detection, fine-tuning, mitigation-plan generation, and the detection prompt-ablation variants. |
| `affect_prompting.py` | Affect/category-aware detection prompts used by the multi-step detection pipelines (sentiment, category). |
| `draw_text.py` | Renders replacement caption text (top/bottom) back onto a mitigated meme. |
| `utils.py` | Lenient parsers for the VLM's JSON responses (detection result, mitigation plan). |

### `src/eval/` — evaluation

| File | Role |
|---|---|
| `run_detection_eval.py` | Standalone detection eval on a JSONL split. Resumable; `--pipeline` selects the prompting strategy. Outputs predictions CSV + metrics. |
| `run_mitigation_eval.py` | Standalone mitigation eval in 3 phases (`--run_mitigation`, `--run_judge`, `--compute_metrics`, or `--all`). |
| `run_experiment_grid.py` | Config-driven grid runner (recommended). Runs any subset of stages (`detection, prompt, diffusion, judge, metrics`) across the models in a YAML config, with N repetitions and aggregation. |
| `experiments.yaml` | Grid config: experiments, models, per-experiment `stages` and optional `pipeline`. |
| `experiments_quick_test.yaml` | Minimal config for a fast smoke test. |

### `src/finetuning/` — fine-tuning

| File | Role |
|---|---|
| `generate_dataset.py` | Builds the synthetic fine-tuning dataset (labels, descriptions, mitigation plans) by querying an LLM API. |
| `format_dataset.py` | Converts `data/finetuning/dataset.jsonl` into task-specific JSONL files (detection / mitigation). |
| `train_detection.py` | LoRA fine-tuning of the VLM for hateful detection. |
| `train_cls_head.py` | Trains a lightweight classification head on top of the frozen VLM (detection as a forward pass). |
| `train_mitigation.py` | QLoRA fine-tuning of the VLM for mitigation-plan generation. |
| `utils_training.py` | Shared helpers for the training scripts. |

### `data/`

JSONL records use the fields `id`, `img` (path relative to the dataset root, e.g. `img/57823.png`), `label` (1 = hateful, 0 = not), and `text` (the meme caption).

| Path | Content |
|---|---|
| `data/hateful-meme/` | Source dataset (Facebook Hateful Memes): `img/` (the `.png` memes), `train.jsonl` / `dev.jsonl` / `test.jsonl`, plus `LICENSE.txt` and `README.md`. Used as `--img_dir`. |
| `data/eval_data/eval_490_balanced.jsonl` | The 490-sample balanced split used for all evaluations (`--jsonl`). |
| `data/finetuning/` | Fine-tuning data produced by `generate_dataset.py` / `format_dataset.py`: `dataset.jsonl` (raw) and the task-specific splits `detection_binary.jsonl`, `detection_full.jsonl`, `detection_with_description.jsonl`, `detection_with_hate_type.jsonl`, `mitigation.jsonl` (see `data/finetuning/README.md`). |

Images (`*.png`) are not tracked in git — point `--img_dir` to wherever the dataset lives (locally or on the cluster).

### `docker/`

| File | Role |
|---|---|
| `Dockerfile` | Builds the CUDA image (PyTorch, transformers, diffusers, eval deps) used on the RCP cluster. |
| `requirements.txt` | Python dependencies installed into the image. |

### `models/`

One folder per fine-tuned model we tested. Each holds the exact `run_config.json` (every training hyperparameter — base model, dataset, LoRA rank, learning rate, epochs, etc.) used to produce that run. The folder name encodes the task and base model: `<task>_<base-model>_<method>`, e.g. `detection_qwen2_5_lora` or `mitigation_gemma4_lora`.

| Prefix | Trained with | Config maps to |
|---|---|---|
| `detection_*_lora` | LoRA detection adapter | `src/finetuning/train_detection.py` |
| `cls_head_*` | Frozen-VLM classification head | `src/finetuning/train_cls_head.py` |
| `mitigation_*_lora` / `mitigation_*_qlora` | (Q)LoRA mitigation adapter | `src/finetuning/train_mitigation.py` |

To reproduce a run, pass the fields from its `run_config.json` to the matching training script. Two examples:

```bash
# detection_qwen2_5_lora — LoRA detection adapter on Qwen2.5-VL-7B
python3 src/finetuning/train_detection.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_jsonl data/finetuning/detection_full.jsonl \
  --output_dir checkpoints/detection_qwen2_5_lora \
  --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 \
  --num_epochs 3 --grad_accum 16 --cache_dir {HF_CACHE_DIR}

# cls_head_gemma4 — classification head on a frozen Gemma-4
python3 src/finetuning/train_cls_head.py \
  --model_name google/gemma-4-31B-it \
  --train_jsonl data/hateful-meme/train.jsonl \
  --dev_jsonl   data/hateful-meme/dev.jsonl \
  --img_dir     data/hateful-meme \
  --output_dir  checkpoints/cls_head_gemma4 \
  --learning_rate 1e-3 --num_epochs 4 --batch_size 4 \
  --balance --cache_dir {HF_CACHE_DIR}

# mitigation_qwen3_6_lora — LoRA mitigation adapter on Qwen3.6-27B
python3 src/finetuning/train_mitigation.py \
  --model_name Qwen/Qwen3.6-27B \
  --dataset_jsonl data/finetuning/mitigation.jsonl \
  --output_dir checkpoints/mitigation_qwen3_6_lora \
  --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 \
  --num_epochs 3 --grad_accum 16 --max_length 2048 --cache_dir {HF_CACHE_DIR}
```


## Usage — commands per use case

All commands assume you run from the repo root, on a GPU node (see the cluster section below). `--cache_dir` points to your Hugging Face cache.

**1. Full mitigation pipeline on a folder of memes**

```bash
python3 src/unhate_pipeline/main.py \
  --vlm_name Qwen/Qwen3.6-27B \
  --diffusion_model_name black-forest-labs/FLUX.2-klein-9B \
  --data_path data/ \
  --cache_dir {HF_CACHE_DIR}
```
Add `--diffusion_offload` on low-VRAM GPUs. Mitigated images are written to `<data_path>/mitigated/`.

**2. Detection evaluation only**

```bash
python3 src/eval/run_detection_eval.py \
  --jsonl   data/eval_data/eval_490_balanced.jsonl \
  --img_dir data/hateful-meme \
  --output  report/detection_predictions.csv \
  --pipeline fewshot_synthetic
```
`--pipeline`: `default | zeroshot | fewshot_synthetic | fewshot_real | sentiment_single | sentiment_chained | category_sentiment | category_fewshot`.

**3. Mitigation evaluation (standalone)**

```bash
python3 src/eval/run_mitigation_eval.py --all \
  --jsonl      data/eval_data/eval_490_balanced.jsonl \
  --img_dir    data/hateful-meme \
  --det_csv    report/detection_predictions.csv \
  --out_dir    report/mitigated \
  --output_csv report/mitigation_results.csv
```

**4. Experiment grid (recommended)**

```bash
# whole pipeline for every experiment in the config
python3 src/eval/run_experiment_grid.py --config src/eval/experiments.yaml \
  --img_dir data/hateful-meme --cache_dir {HF_CACHE_DIR}

# run only some eval types
python3 src/eval/run_experiment_grid.py --config src/eval/experiments.yaml \
 --eval_types baseline --img_dir data/hateful-meme 

# quick smoke test on 5 images
python3 src/eval/run_experiment_grid.py --config src/eval/experiments_quick_test.yaml \
  --img_dir data/hateful-meme --output_root report/_smoke --limit 5
```
Stages: `detection`, `prompt` (diffusion-prompt generation), `diffusion`, `judge`, `metrics`. `judge` runs automatically before `metrics` when needed. Results: `<output_root>/summary.csv` and `summary_aggregated.csv`.

**5. Fine-tuning**

```bash
# build then format the synthetic dataset (generate_dataset needs ANTHROPIC_API_KEY)
python3 src/finetuning/generate_dataset.py
python3 src/finetuning/format_dataset.py

# LoRA detection adapter
python3 src/finetuning/train_detection.py \
  --dataset_jsonl data/finetuning/detection_full.jsonl \
  --output_dir checkpoints/detect/gemma4 --cache_dir {HF_CACHE_DIR}

# Classification head (detection as a forward pass)
python3 src/finetuning/train_cls_head.py \
  --train_jsonl data/finetuning/detection_binary.jsonl \
  --dev_jsonl   data/eval_data/eval_490_balanced.jsonl \
  --img_dir     data/hateful-meme \
  --output_dir  checkpoints/cls_head/gemma4 --cache_dir {HF_CACHE_DIR}

# QLoRA mitigation adapter
python3 src/finetuning/train_mitigation.py \
  --dataset_jsonl data/finetuning/mitigation.jsonl \
  --output_dir checkpoints/mitigate/gemma4 --cache_dir {HF_CACHE_DIR}
```

Pass the resulting checkpoints back via `--adapter_path`, `--cls_head_path` or `--mitigation_adapter` in the pipeline / eval scripts (e.g. in `experiments.yaml`).


## Running on the EPFL RCP cluster

## Build the Docker image

Install Docker on your local machine if you haven't already. Then, go in the docker/ directory. If you want to add dependecies, you can modify the `requirements.txt` file and rebuild the image.

Once done, you can build the image using the following command:

```bash
docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v{VERSION}
```

Then, push the image to the registry:

```bash
docker push registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v{VERSION}
```


## Connect to the cluster and run scripts

To connect to the cluster, you can use the following command:

```bash
ssh {EPFL_USERNAME}@jumphost.rcp.epfl.ch:/mnt/course-ee-559/rcp-caas-ee-559-g09/scratch-g09
```

Once connected, go to the group directory:

```bash
cd /mnt/course-ee-559/rcp-caas-ee-559-g09/scratch-g09/UnHateMemeDL
```

Then, you to be able to run the scripts, lets request a node with GPU:

- For interactive session:

```bash
runai submit \
  --name unhatememe \
  --run-as-uid {uid} \
  --image registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v{VERSION} \
  --gpu 1 \
  --node-pools default \
  --existing-pvc claimname=course-ee-559-scratch-g09,path=/scratch-g09 \
  --existing-pvc claimname=home,path=/home/{EPFL_USERNAME} \
  --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
  --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
  --interactive --attach
```

Note: replace `{uid}` with your user id, which you can get using the `id -u` command. {VERSION} should be replaced with the version of the image you built and pushed to the registry (if you didn't build and push the image, you can use the `1.1` tag). {EPFL_USERNAME} should be replaced with your EPFL username.

- For running a script inside the container, example for inference:

```bash
python3 UnHateMemeDL/src/unhate_pipeline/main.py \
  --vlm_name {VLM_NAME} \
  --data_path {DATA_PATH} \
  --cache_dir {HF_CACHE_DIR}
```

Note: replace `{VLM_NAME}` with the name of the Vision-Language Model you want to use (e.g., "Qwen/Qwen3.6-27B") should be correspond to the model name in the Hugging Face Hub. `{DATA_PATH}` should be replaced with the path to the dataset you want to use for inference. `{HF_CACHE_DIR}` should be replaced with the path to the Hugging Face cache directory where the models will be downloaded and stored.

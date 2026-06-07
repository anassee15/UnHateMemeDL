# UnHateMemeDL

Mitigate hateful content on image meme using open source Vision-Language Models (VLMs) and Diffusion Models.

## Detection prompting pipelines

The hateful meme detection step (`src/eval/run_detection_eval.py`) supports several prompting strategies via `--pipeline`:

| Pipeline | Description | VLM calls |
|---|---|---|
| `default` **(default)** | Adapter-aware generative detection (the baseline behavior; uses the fine-tuning prompt when a detection LoRA adapter is active, otherwise `HATEFUL_DETECTION_PROMPT`). | 1 |
| `zeroshot` | Hate definition + classification criteria only. No examples, no affect routing. | 1 |
| `fewshot_synthetic` | Adds 17 synthetic calibration examples spanning the full range of hate types (explicit, implicit, culturally coded, historical, intersectional). Raises model confidence for borderline cases. | 1 |
| `fewshot_real` | Adds 4 real labeled examples from the eval set (IDs 80243, 9467, 62375, 91756 — **must be excluded from metrics**). | 1 |
| `sentiment_single` | Single-call affect-aware detection. The model reasons over sentiment, humor, sarcasm, and offensiveness internally before classifying. | 1 |
| `sentiment_chained` | Two-step chain: (1) classify affect/sentiment dimensions, (2) hate detection conditioned on the detected affect profile. | 2 |
| `category_sentiment` | Two-to-three-step chain: (1) classify meme category (historical / general\_culture / identity\_social), (2) classify affect if non-historical, (3) hate detection conditioned on category + affect. | 2–3 |
| `category_fewshot` | Two-step chain: (1) classify meme category, (2) hate detection with category-specific few-shot examples. | 2 |

These prompts live in `src/unhate_pipeline/prompt.py` and `src/unhate_pipeline/affect_prompting.py`. Quantitative comparison across pipelines: see [`report/pipeline_comparison.md`](report/pipeline_comparison.md) and [`report/detection_results.md`](report/detection_results.md).

Example usage:

```bash
python src/eval/run_detection_eval.py \
    --jsonl  data/eval_data/eval_490_balanced.jsonl \
    --img_dir data/eval_data \
    --output  report/detection_predictions.csv \
    --pipeline fewshot_synthetic
```

The `--pipeline` choice applies to the generative path only; it is ignored when `--cls_head_path` (classification-head detection) is set.

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

Note: replace `{uid}` with your user id, which you can get using the `id -u` command. {VERSION} should be replaced with the version of the image you built and pushed to the registry (if you didn't build and push the image, you can use the `1.0` tag). {EPFL_USERNAME} should be replaced with your EPFL username.

- For running a script inside the container, example for inference:

```bash
python3 UnHateMemeDL/src/unhate_pipeline/main.py \
  --vlm_name {VLM_NAME} \
  --data_path {DATA_PATH} \
  --cache_dir {HF_CACHE_DIR}
```

Note: replace `{VLM_NAME}` with the name of the Vision-Language Model you want to use (e.g., "Qwen/Qwen3.6-27B") should be correspond to the model name in the Hugging Face Hub. `{DATA_PATH}` should be replaced with the path to the dataset you want to use for inference. `{HF_CACHE_DIR}` should be replaced with the path to the Hugging Face cache directory where the models will be downloaded and stored.

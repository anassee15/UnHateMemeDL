# Experiment Grid Evaluation — Quick Start Guide

Multi-experiment grid runner with **N repetitions** pour évaluation statistique.

## Installation

```bash
pip install pyyaml
```

## Configuration — `experiments.yaml`

Le fichier `experiments.yaml` définit tous les experiments à lancer.

### Structure

```yaml
n_reps: 3  # Nombre de répétitions par (eval_type, modèle)

experiments:
  eval_type_name:
    description: "..."
    run_mitigation: true/false
    models:
      - google/gemma-4-31B-it                    # Simple: juste le model name
      - model: Qwen/Qwen3.6-27B
        cls_head_path: path/to/classifier.pt    # Avec options
      - model: Qwen/Qwen2.5-VL-7B-Instruct
        adapter_path: path/to/lora_adapter      # LoRA detection
        mitigation_adapter_path: path/to/lora   # LoRA mitigation
```

### Exemples

**Baseline (aucun adapter)** :
```yaml
baseline:
  description: "Baseline detection + mitigation"
  run_mitigation: true
  models:
    - google/gemma-4-31B-it
    - Qwen/Qwen3.6-27B
    - Qwen/Qwen2.5-VL-7B-Instruct
```

**Classification Head** :
```yaml
cls_head:
  description: "Detection avec classification head"
  run_mitigation: false
  models:
    - model: google/gemma-4-31B-it
      cls_head_path: checkpoints/cls_head/gemma4/best_classifier.pt
    - model: Qwen/Qwen2.5-VL-7B-Instruct
      cls_head_path: checkpoints/cls_head/qwen25/best_classifier.pt
```

**LoRA Detection** :
```yaml
lora_detect:
  description: "Detection avec LoRA adapter"
  run_mitigation: false
  models:
    - model: google/gemma-4-31B-it
      adapter_path: checkpoints/detect/gemma4
    - model: Qwen/Qwen2.5-VL-7B-Instruct
      adapter_path: checkpoints/detect/qwen25
```

**LoRA Detection + Mitigation** :
```yaml
lora_mitigate:
  description: "Detection + mitigation avec LoRA adapters"
  run_mitigation: true
  models:
    - model: google/gemma-4-31B-it
      adapter_path: checkpoints/detect/gemma4
      mitigation_adapter_path: checkpoints/mitigate/gemma4
    - model: Qwen/Qwen2.5-VL-7B-Instruct
      adapter_path: checkpoints/detect/qwen25
      mitigation_adapter_path: checkpoints/mitigate/qwen25
```

## Utilisation

### 1. Smoke test (5 min)
```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types baseline \
  --n_reps 1 \
  --skip_mitigation
```

### 2. Lancer tous les experiments
```bash
python run_experiment_grid.py --config experiments.yaml
```

### 3. Filtrer par eval_type
```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types baseline cls_head lora_detect
```

### 4. Override n_reps
```bash
python run_experiment_grid.py --config experiments.yaml \
  --n_reps 5
```

### 5. Détection seulement (skip mitigation)
```bash
python run_experiment_grid.py --config experiments.yaml \
  --skip_mitigation
```

### 6. Recompute metrics (no GPU)
```bash
python run_experiment_grid.py --config experiments.yaml \
  --metrics_only
```

### 7. Réagréger les résultats (print table + CSV)
```bash
python run_experiment_grid.py --config experiments.yaml \
  --aggregate_only
```

## Résultats

```
report/full_eval/
├── baseline_google_gemma-4-31B-it/
│   ├── rep_00/
│   │   ├── detection_predictions.csv
│   │   ├── mitigation_results.csv
│   │   └── mitigated/
│   │       ├── {id}_mitigated.png
│   │       └── {id}_intermediate.json
│   ├── rep_01/
│   └── rep_02/
├── cls_head_google_gemma-4-31B-it/
│   └── rep_00/ ...
├── lora_detect_google_gemma-4-31B-it/
│   └── rep_00/ ...
├── lora_mitigate_google_gemma-4-31B-it/
│   └── rep_00/ ...
├── summary.csv  ← Toutes les runs: eval_type, rep, vlm_name, model_slug, + 32 metrics
└── summary_aggregated.csv  ← Mean±std par (eval_type, model)
```

### Colonnes de `summary.csv`
- `eval_type` : baseline, cls_head, lora_detect, lora_mitigate
- `rep` : 0, 1, 2 (repetition index)
- `vlm_name` : model HF ID
- `model_slug` : filesystem-safe slug
- Detection metrics : `auroc`, `macro_f1`, `accuracy`, `tp`, `tn`, `fp`, `fn`, timing stats
- Mitigation metrics : `mean_prob_before`, `mean_prob_after`, `mean_tr_pct`, `mean_bertscore_f1`, `mean_clip_score`, `mean_ssim`, `mean_mps`, timing stats

### Table d'agrégation (`--aggregate_only`)
Affiche une table ASCII avec `mean_auroc±std_auroc`, `mean_macro_f1±std_macro_f1`, etc. :
```
eval_type            vlm_name             n_reps               auroc                macro_f1             ...
baseline             google/gemma-4-31B-it  3                0.9234±0.0045        0.8765±0.0123        ...
baseline             Qwen/Qwen3.6-27B       3                0.8956±0.0067        0.8234±0.0089        ...
cls_head             google/gemma-4-31B-it  3                0.9123±0.0000        0.8654±0.0000        ...
```

## Resumability

Chaque phase est **resumable** — relancer le même command reprend où ça s'est arrêté:

- **Detection CSV** : images déjà traitées sont skipped (keyed on `id`)
- **Mitigation JSON** : prompts déjà générés sont skipped (fichiers existent)
- **Summary CSV** : rows déjà présentes (keyed on `eval_type, model_slug, rep`) sont skipped

Si un run crash au milieu, relance simplement la même commande.

## Optimisation mémoire

La pipeline divise la mitigation en **3 phases** pour éviter d'avoir VLM + diffusion en GPU simultanément :

```
Phase A (VLM only):     Detection + prompt generation → _intermediate.json
Phase B (Diffusion):    Apply diffusion using JSONs → _mitigated.png
Phase C (VLM only):     Judge + compute metrics
```

Peak GPU memory : `max(VLM, diffusion)` ≈ 62GB (vs. 92GB without splitting)

## Custom configs

Pour créer une variante (ex: test rapide) :

```bash
cp experiments.yaml experiments_test.yaml
```

Puis édite `experiments_test.yaml` :
```yaml
n_reps: 1  # Juste une rep

experiments:
  baseline:
    run_mitigation: false  # Pas de mitigation
    models:
      - google/gemma-4-31B-it  # Un seul modèle
```

Lancer avec :
```bash
python run_experiment_grid.py --config experiments_test.yaml
```

## Troubleshooting

**`PyYAML not installed`**
```bash
pip install pyyaml
```

**`Config file not found`**
```bash
# Assure-toi que le chemin est correct
python run_experiment_grid.py --config src/eval/experiments.yaml
```

**OOM (Out of Memory)**
- Réduis le nombre de reps : `--n_reps 1`
- Lancer seulement la détection : `--skip_mitigation`
- Vérifier que les checkpoints paths existent

**Partial run crashed**
- Relance la même commande — ça reprend automatiquement
- Ou utilise `--metrics_only` pour recomputer sans GPU

## Ajouter un nouvel eval_type

1. Édite `experiments.yaml`
2. Ajoute une nouvelle section (ex: `few_shot`)
3. Spécifie les modèles et options
4. Lancer : `python run_experiment_grid.py --config experiments.yaml --eval_types few_shot`

Exemple :
```yaml
few_shot:
  description: "Few-shot detection avec K examples"
  run_mitigation: false
  models:
    - google/gemma-4-31B-it
    - Qwen/Qwen3.6-27B
```

# Commands to Run Each Experiment

Copy-paste ready commands for each experiment type.

**Prerequisites:**
```bash
pip install pyyaml
cd src/eval
```

---

## 🚀 Quick Test (5 min)
```bash
python run_experiment_grid.py --config experiments_quick_test.yaml
```

---

## 📋 Full Experiments

### 1️⃣ Baseline (Detection + Mitigation)
All 3 models, 3 reps each, ~2-3 hours

```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types baseline \
  --n_reps 3
```

---

### 2️⃣ Classification Head (Detection Only)
All 3 models, 3 reps, ~30-40 min

```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types cls_head \
  --n_reps 3
```

---

### 3️⃣ LoRA Detection (Detection Only)
Gemma4 + Qwen2.5, 3 reps, ~20 min

```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types lora_detect \
  --n_reps 3
```

---

### 4️⃣ LoRA Mitigation (Detection + Mitigation)
Gemma4 + Qwen2.5, 3 reps, ~1-1.5 hours

```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types lora_mitigate \
  --n_reps 3
```

---

## ⚡ Fast Variants (Mitigation Only)

### Baseline Mitigation Only
Hateful images only, 3 reps, ~45 min

```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types baseline \
  --mitigation_only \
  --hateful_only \
  --n_reps 3
```

---

### LoRA Mitigation Only
Hateful images only, 3 reps, ~30 min

```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types lora_mitigate \
  --mitigation_only \
  --hateful_only \
  --n_reps 3
```

---

## 🔍 Individual Model Tests

### Baseline - Gemma4 Only
```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types baseline \
  --n_reps 1
# Then edit experiments.yaml to keep only google/gemma-4-31B-it
```

Or create a custom YAML with single model:
```bash
cat > baseline_gemma4.yaml << 'EOF'
n_reps: 3
experiments:
  baseline:
    description: "Baseline - Gemma4 only"
    run_mitigation: true
    models:
      - google/gemma-4-31B-it
EOF

python run_experiment_grid.py --config baseline_gemma4.yaml
```

---

## 📊 View Results

### After all experiments:
```bash
python run_experiment_grid.py --config experiments.yaml --aggregate_only
```

Shows table with mean±std for each experiment/model combination.

---

## 📝 Check Progress

Results accumulate in `report/full_eval/`:
```bash
# Count completed runs
ls -R report/full_eval/ | grep "detection_predictions.csv" | wc -l

# Check summary
head -20 report/full_eval/summary.csv
```

---

## 🔄 Resume Interrupted Runs

Just re-run the same command — it picks up where it left off:
```bash
# If interrupted, just repeat:
python run_experiment_grid.py --config experiments.yaml --eval_types baseline --n_reps 3
```

---

## 💡 Tips

**Start with baseline detection only (fast):**
```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types baseline \
  --skip_mitigation \
  --n_reps 1
```

**Test mitigation pipeline without detection (faster):**
```bash
python run_experiment_grid.py --config experiments.yaml \
  --eval_types lora_mitigate \
  --mitigation_only \
  --hateful_only \
  --n_reps 1
```

**Recompute metrics from existing CSVs (no GPU):**
```bash
python run_experiment_grid.py --config experiments.yaml --metrics_only
```

**Use custom data:**
```bash
python run_experiment_grid.py --config experiments.yaml \
  --jsonl data/hateful-meme/dev.jsonl \
  --eval_types baseline \
  --n_reps 1
```

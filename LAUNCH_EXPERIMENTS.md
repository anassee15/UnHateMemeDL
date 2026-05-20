# 🚀 Commands to Launch Each Experiment

**Setup (once):**
```bash
cd src/eval
pip install pyyaml
```

---

## 1️⃣ Baseline (Detection + Mitigation)
All 3 models, 3 reps, ~2-3 hours

```bash
python run_experiment_grid.py --config experiments.yaml --eval_types baseline --n_reps 3
```

---

## 2️⃣ Classification Head (Detection Only)
All 3 models, 3 reps, ~30-40 min

```bash
python run_experiment_grid.py --config experiments.yaml --eval_types cls_head --n_reps 3
```

---

## 3️⃣ LoRA Detection (Detection Only)
Gemma4 + Qwen2.5, 3 reps, ~20 min

```bash
python run_experiment_grid.py --config experiments.yaml --eval_types lora_detect --n_reps 3
```

---

## 4️⃣ LoRA Mitigate (Mitigation Only)
Gemma4 + Qwen2.5, 3 reps, ~45 min
*(skips detection automatically via YAML config)*

```bash
python run_experiment_grid.py --config experiments.yaml --eval_types lora_mitigate --n_reps 3
```

---

## 📊 View Results

After all experiments:

```bash
python run_experiment_grid.py --config experiments.yaml --aggregate_only
```

Shows table with **mean±std** for each experiment/model.

---

## 🎯 Quick Test First

```bash
python run_experiment_grid.py --config experiments_quick_test.yaml
```

---

## 💡 Run in Sequence

```bash
# 1. Test
python run_experiment_grid.py --config experiments_quick_test.yaml

# 2. Baseline
python run_experiment_grid.py --config experiments.yaml --eval_types baseline --n_reps 3

# 3. Classification Head
python run_experiment_grid.py --config experiments.yaml --eval_types cls_head --n_reps 3

# 4. LoRA Detection
python run_experiment_grid.py --config experiments.yaml --eval_types lora_detect --n_reps 3

# 5. LoRA Mitigate
python run_experiment_grid.py --config experiments.yaml --eval_types lora_mitigate --n_reps 3

# 6. View Results
python run_experiment_grid.py --config experiments.yaml --aggregate_only
```

---

## 📁 Results Location

All results go to: `report/full_eval/`

- `summary.csv` — all runs
- `summary_aggregated.csv` — mean±std per experiment
- `baseline_*/`, `cls_head_*/`, `lora_detect_*/`, `lora_mitigate_*/` — per-experiment directories

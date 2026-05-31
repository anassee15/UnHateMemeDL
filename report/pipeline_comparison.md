# Detection Pipeline Comparison

Evaluated on `data/eval_data/eval_490_balanced.jsonl` — 490 memes, balanced 245 hateful / 245 non-hateful.
Prec / Rec / F1 are for the hateful class (label=1). Model: Qwen/Qwen3.6-27B.

## Main metrics

| Pipeline | AUROC | Macro-F1 | Accuracy | Prec (hat) | Rec (hat) | F1 (hat) |
|---|---|---|---|---|---|---|
| `fewshot_synthetic` | **0.7552** | **0.7102** | **0.7102** | 0.7102 | 0.7102 | 0.7102 |
| `fewshot_real` | 0.7493 | 0.6723 | 0.6796 | 0.6384 | 0.8286 | 0.7211 |
| `category_fewshot` | 0.7208 | 0.6621 | 0.6735 | 0.6269 | **0.8571** | **0.7241** |
| `category_sentiment` | 0.6609 | 0.6459 | 0.6469 | 0.6324 | 0.7020 | 0.6654 |
| `sentiment_single` | 0.6475 | 0.6244 | 0.6245 | 0.6215 | 0.6367 | 0.6290 |
| `sentiment_chained` | 0.6344 | 0.6116 | 0.6122 | 0.6038 | 0.6531 | 0.6275 |

## Confusion matrix

| Pipeline | TP | TN | FP | FN | FP% | FN% |
|---|---|---|---|---|---|---|
| `fewshot_synthetic` | 174 | 174 | 71 | 71 | 29.0 | 29.0 |
| `fewshot_real` | 203 | 130 | 115 | 42 | 46.9 | 17.1 |
| `category_fewshot` | 210 | 120 | 125 | 35 | 51.0 | **14.3** |
| `category_sentiment` | 172 | 145 | 100 | 73 | 40.8 | 29.8 |
| `sentiment_single` | 156 | 150 | 95 | 89 | 38.8 | 36.3 |
| `sentiment_chained` | 160 | 140 | 105 | 85 | 42.9 | 34.7 |

## Key observations

- **`fewshot_synthetic` is the strongest overall.** It wins on AUROC, Macro-F1, and accuracy, and is the only pipeline where FP and FN are perfectly symmetric (71/71) — reflecting genuinely balanced classification behaviour rather than a bias toward either class.

- **`fewshot_real` and `category_fewshot` prioritise recall at the cost of precision.** Both miss very few hateful memes (FN 17% and 14%) but flag roughly half of all non-hateful memes as hateful (FP 47% and 51%). Whether that trade-off is acceptable depends on the downstream use case.

- **The sentiment pipelines underperform across every metric.** `sentiment_single`, `sentiment_chained`, and `category_sentiment` all fall below 0.67 on Macro-F1 and AUROC. The affect-conditioning appears to add noise rather than signal for this model and dataset.

- **`sentiment_chained` is the weakest pipeline despite using the most reasoning steps** (AUROC 0.6344). The intermediate affect classification step appears to mislead the second hate-detection call rather than helping it.

- **Category routing alone (`category_fewshot`) recovers most of the few-shot gain** without the affect step, suggesting that category context is the useful signal in the `category_sentiment` pipeline, not the affect labels.

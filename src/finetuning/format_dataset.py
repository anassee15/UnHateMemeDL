"""
Format data/finetuning/dataset.jsonl into task-specific JSONL files.

Outputs (all in data/finetuning/):
  detection_binary.jsonl          — {classification}
  detection_with_description.jsonl — {classification, probability, description}
  detection_with_hate_type.jsonl  — + hate_type
  detection_full.jsonl            — + hate_location
  mitigation.jsonl                — hateful-only, 5-field mitigation target

Usage:
    python src/finetuning/format_dataset.py \
        [--dataset    data/finetuning/dataset.jsonl] \
        [--exclude    data/eval_data/eval_490_balanced.jsonl] \
        [--output_dir data/finetuning]
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_exclude_ids(exclude_jsonl: Optional[str]) -> set:
    if not exclude_jsonl:
        return set()
    return {f"hm_{item['id']}" for item in load_jsonl(exclude_jsonl) if "id" in item}


def stats(records: list[dict]) -> str:
    hateful = sum(1 for r in records if r.get("label") == 1)
    non_hateful = sum(1 for r in records if r.get("label") == 0)
    unlabeled = len(records) - hateful - non_hateful
    parts = [f"{len(records)} total", f"{hateful} hateful", f"{non_hateful} non-hateful"]
    if unlabeled:
        parts.append(f"{unlabeled} unlabeled")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/finetuning/dataset.jsonl")
    parser.add_argument("--exclude", default="data/eval_data/eval_490_balanced.jsonl")
    parser.add_argument("--output_dir", default="data/finetuning")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exclude_ids = build_exclude_ids(args.exclude)
    source = load_jsonl(args.dataset)
    print(f"Loaded {len(source)} records from {args.dataset}")
    print(f"Excluding {len(exclude_ids)} eval IDs\n")

    detection_binary = []
    detection_description = []
    detection_hate_type = []
    detection_full = []
    mitigation = []

    for rec in source:
        if rec.get("id") in exclude_ids:
            continue

        label = rec.get("ground_truth_label")
        classification = "hateful" if label == 1 else "non-hateful"
        base = {
            "id": rec["id"],
            "img_path": rec["img_path"],
            "text": rec.get("text", "") or "",
            "label": label,
        }

        detection_binary.append({**base, "target": {"classification": classification}})

        detection_description.append({
            **base,
            "target": {
                "classification": classification,
                "probability": rec.get("probability"),
                "description": rec.get("description", ""),
            },
        })

        detection_hate_type.append({
            **base,
            "target": {
                "classification": classification,
                "probability": rec.get("probability"),
                "description": rec.get("description", ""),
                "hate_type": rec.get("hate_type"),
            },
        })

        detection_full.append({
            **base,
            "target": {
                "classification": classification,
                "probability": rec.get("probability"),
                "description": rec.get("description", ""),
                "hate_type": rec.get("hate_type"),
                "hate_location": rec.get("hate_location"),
            },
        })

        if label == 1:
            mit = rec.get("mitigation") or {}
            diffusion_prompt = mit.get("diffusion_prompt", "")
            if isinstance(diffusion_prompt, str) and len(diffusion_prompt.strip()) >= 10:
                mitigation.append({
                    "id": rec["id"],
                    "img_path": rec["img_path"],
                    "text": rec.get("text", "") or "",
                    "target": {
                        "hate_source": rec.get("description", ""),
                        "hate_location": rec.get("hate_location"),
                        "diffusion_prompt": diffusion_prompt,
                        "original_text": mit.get("original_text"),
                        "replacement_text": mit.get("replacement_text"),
                    },
                })

    outputs = [
        ("detection_binary.jsonl", detection_binary),
        ("detection_with_description.jsonl", detection_description),
        ("detection_with_hate_type.jsonl", detection_hate_type),
        ("detection_full.jsonl", detection_full),
        ("mitigation.jsonl", mitigation),
    ]

    for fname, records in outputs:
        path = out / fname
        write_jsonl(path, records)
        print(f"{fname}: {stats(records)}")


if __name__ == "__main__":
    main()

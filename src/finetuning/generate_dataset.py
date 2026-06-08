import anthropic
import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

SYSTEM_PROMPT = """## Role
You are a dataset curator for a multimodal hate speech detection and mitigation system.

Given a meme image, its text, and its ground-truth label, return a structured JSON annotation.
Your classification MUST match the provided label.
Prioritize: mitigation and description of the hateful content.

## Output
Return ONLY one valid JSON object with exactly this structure:

{
  "meme_category": {
    "label": "historical" | "general_culture" | "identity_social",
    "probability": <float 0.0-1.0>
  },
  "affective_register": {
    "overall_sentiment": {
      "label": "negative" | "neutral" | "positive",
      "probability": <float 0.0-1.0>
    },
    "humour": {
      "label": "not_funny" | "funny" | "very_funny" | "hilarious",
      "probability": <float 0.0-1.0>
    },
    "sarcasm": {
      "label": "not_sarcastic" | "general_sarcasm" | "twisted_meaning" | "very_twisted",
      "probability": <float 0.0-1.0>
    },
    "offense": {
      "label": "not_offensive" | "slightly_offensive" | "very_offensive" | "highly_offensive",
      "probability": <float 0.0-1.0>
    },
    "motivation": {
      "label": "not_motivational" | "motivational",
      "probability": <float 0.0-1.0>
    }
  },
  "classification": "hateful" | "not_hateful",
  "probability": <float 0.0-1.0>,
  "hate_type": null | ["racist" | "xenophobic" | "religious_hate" | "antisemitic" | "islamophobic" | "sexist" | "homophobic" | "transphobic" | "ableist" | "anti_immigrant" | "historical_violence" | "dehumanization" | "other"],
  "hate_location": null | "VISUAL_ONLY" | "TEXT_ONLY" | "COMBINED" | "INTERSECTIONAL",
  "description": "<one sentence explaining precisely why the meme is hateful or not>",
  "mitigation": {
    "diffusion_prompt": "<2-4 sentences, visual-only description of the mitigated scene; never mention text or overlays>",
    "original_text": "<exact meme text, or null>",
    "replacement_text": "<rewritten meme text removing hate while preserving the joke/point, or null; at most one \\n>",
    "mitigation_strategy": "<one sentence explaining the change>"
  }
}

## Rules

### 1) Meme category
Choose exactly one:
- historical: mainly about historical events, figures, wars, genocide, colonialism, slavery, Nazism, fascism, terrorism, political atrocities, or other historical violence.
- general_culture: mainly about everyday life, internet culture, work, school, relationships, entertainment, absurd humor, general politics, or non-identity social situations.
- identity_social: mainly about race, ethnicity, religion, nationality, immigration, gender, sexual orientation, disability, or other protected/social identity.

Probability here = confidence in the chosen category.

### 2) Affective register
Use the given labels only.
Each probability = confidence in that affective label.
When sarcasm exists, judge implied meaning, not just literal wording.
Humor can still be hateful if it degrades a protected group.

### 3) Classification
Classification MUST match the ground-truth label.
Main probability = confidence in hateful vs not_hateful.
Guideline:
- < 0.3 = clearly benign
- > 0.7 = clearly hateful
- 0.3-0.6 = ambiguous

A meme is hateful if it attacks, dehumanizes, threatens, excludes, humiliates, endorses harm against, or spreads hate toward a protected/identity-based group. Protected or identity-based groups include race, ethnicity, religion, nationality, immigration status, gender, sexual orientation, disability, and similar social identity categories.

### 4) Historical memes
A historical meme is hateful only if it mocks victims, glorifies perpetrators, denies/minimizes atrocities, promotes supremacist ideology, or uses historical violence to target a protected group.
Mere historical reference, criticism of history, or criticism of historical figures is not enough.
If historical violence is used to degrade or threaten a group, use "historical_violence".

### 5) hate_type
- null if classification = "not_hateful"
- one or more labels if classification = "hateful"

Use:
- racist: race/ethnicity
- xenophobic: nationality/foreignness/national origin
- religious_hate: religion in general
- antisemitic: Jewish people
- islamophobic: Muslims
- sexist: sex/gender
- homophobic: gay/lesbian/same-sex attracted people
- transphobic: transgender/gender-diverse people
- ableist: disabled people
- anti_immigrant: immigrants/migrants/refugees
- historical_violence: genocide, slavery, colonial violence, Nazism, fascism, terrorism, mass killing, or similar atrocity
- dehumanization: compares people to animals, disease, filth, parasites, objects, etc.
- other: hateful but uncategorized

### 6) hate_location

- VISUAL_ONLY: image hateful, text neutral
- TEXT_ONLY: text hateful, image benign
- COMBINED: hate emerges from image + text together
- INTERSECTIONAL: both image and text are independently hateful
- null if not_hateful

### 7) description

Write exactly one sentence explaining precisely why the meme is hateful or not.

- Be specific and factual.
- Identify the target group if there is one, or explicitly indicate that no protected group is targeted.
- State the harmful mechanism when relevant: stereotype, insult, humiliation, exclusion, threat, dehumanization, endorsement of harm, or glorification/minimization of historical atrocity.
- Mention whether the meaning comes from the image, the text, or both when relevant.
- For hateful memes, explain what makes the content hateful toward a protected group.
- For not_hateful memes, explain why the meme does not express hate toward a protected group, even if it is offensive, dark, or sarcastic.
- Use implied meaning when sarcasm or irony is present.
- Do not mention probabilities, uncertainty, mitigation, or the annotation process.
- Do not use vague statements like "it is hateful" without justification.
- Keep it to one concise sentence.


### 8) Mitigation

Preserve as much as possible: semantic content, intent, joke structure, humor, meme format, and communicative function.
Change only what is necessary to remove hate.

Rules by case:
- VISUAL_ONLY: change visual only; replacement_text = null if text is already neutral
- TEXT_ONLY: keep scene unchanged; only rewrite text
- COMBINED / INTERSECTIONAL: diffusion_prompt fixes visual element only; replacement_text fixes text only
- not_hateful: diffusion_prompt faithfully reproduces the original scene; replacement_text = null

diffusion_prompt:
- 2-4 sentences
- visual-only
- describe how the meme should be changed to remove the hateful content by a diffusion model
- describe scene, lighting, style, composition
- never mention text, captions, words, slogans, writing, letters, typography, subtitles, or overlays

original_text:
- exact meme text if present, else null

replacement_text:
- remove hateful targeting while preserving joke, point, tone, or format when possible
- null if original text is already non-hateful

mitigation_strategy:
- one sentence explaining what changed and why

Return ONLY the JSON object, no markdown fences, no explanation."""


def load_eval_ids(data_dir: Path) -> set[str]:
    """Return the set of record IDs present in any eval_data jsonl file."""
    eval_ids = set()
    eval_dir = data_dir / "eval_data"
    if not eval_dir.exists():
        print(f"Warning: eval_data directory not found at {eval_dir}. No eval records will be excluded.")
        return eval_ids
    for path in eval_dir.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    eval_ids.add(f"hm_{json.loads(line)['id']}")
                except Exception:
                    pass
    return eval_ids


def load_hateful_memes(data_dir: Path, exclude_ids: set[str] | None = None) -> list[dict]:
    records = []
    for split in ["train.jsonl", "dev.jsonl"]:
        path = data_dir / "hateful-meme" / split
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                record_id = f"hm_{row['id']}"
                if exclude_ids and record_id in exclude_ids:
                    continue
                img_path = data_dir / "hateful-meme" / row["img"]
                if img_path.exists():
                    records.append({
                        "id": record_id,
                        "img_path": str(img_path),
                        "text": row.get("text", ""),
                        "label": row["label"],
                        "source": "hateful_memes",
                    })
    return records


def sample_phase(records: list[dict], label: int, target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = [r for r in records if r["label"] == label]
    rng.shuffle(pool)
    return pool[:target]


def image_source(img_path: str) -> dict:
    suffix = Path(img_path).suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

    with open(img_path, "rb") as f:
        b64, media_type = base64.standard_b64encode(f.read()).decode("utf-8"), media_type

    return {"type": "base64", "media_type": media_type, "data": b64}


def user_text(text: str, label: int) -> str:
    label_str = "hateful" if label == 1 else "not_hateful"
    return f'Meme text: "{text}"\nLabel: {label_str}\n\nAnalyze this meme and return the JSON output as instructed.'


def call_claude(
    client: anthropic.Anthropic,
    record: dict,
    usage_totals: dict,
) -> tuple[dict | None, dict | None]:
    """Returns (parsed_result, per_call_usage) or (None, None) on failure."""
    try:
        actual_source = image_source(record["img_path"])
    except Exception as e:
        print(f"  [encode error] {e}")
        return None, None

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": actual_source},
            {"type": "text", "text": user_text(record["text"], record["label"])},
        ],
    }]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )

        u = response.usage
        call_usage = {
            "input": u.input_tokens,
            "output": u.output_tokens,
            "cache_creation": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        for k, v in call_usage.items():
            usage_totals[k] += v

        if response.stop_reason == "refusal" or not response.content:
            print(f"  [refusal] model refused (stop_reason={response.stop_reason})")
            return None, call_usage

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json")
        return json.loads(raw), call_usage
    except json.JSONDecodeError as e:
        print(f"  [json error] {e}")
        return None, None
    except anthropic.RateLimitError:
        print("  [rate limit] sleeping 60s...")
        time.sleep(60)
        return None, None
    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            print("  [overloaded] sleeping 30s...")
            time.sleep(30)
            return None, None
        print(f"  [api error {e.status_code}] {e}")
        return None, None
    except anthropic.APIError as e:
        print(f"  [api error] {e}")
        return None, None


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def count_done_by_label(output_path: Path) -> tuple[int, int]:
    """Returns (done_hateful, done_not_hateful) from the output file."""
    if not output_path.exists():
        return 0, 0
    hateful, not_hateful = 0, 0
    with open(output_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("ground_truth_label") == 1:
                    hateful += 1
                else:
                    not_hateful += 1
            except Exception:
                pass
    return hateful, not_hateful


def check_credit_balance(client: anthropic.Anthropic, min_balance_usd: float) -> tuple[bool, float | None]:
    """
    Try to check Anthropic credit balance via the API.
    Returns (has_sufficient_balance, balance_usd).
    If balance cannot be determined, returns (True, None) and caller should warn.
    """
    try:
        response = client._client.get("/v1/account/credits")
        if response.status_code == 200:
            data = response.json()
            for key in ["available_credits", "credits_remaining", "balance", "total_credits"]:
                if key in data:
                    bal_usd = float(data[key]) / 100
                    return bal_usd >= min_balance_usd, bal_usd
    except Exception:
        pass
    return True, None


def _compute_cost(usage: dict) -> float:
    # Sonnet 4.6 pricing (per 1M tokens): input $3, output $15, cache_write $3.75, cache_read $0.30
    return (
        usage["input"] * 3.00 / 1_000_000
        + usage["output"] * 15.00 / 1_000_000
        + usage["cache_creation"] * 3.75 / 1_000_000
        + usage["cache_read"] * 0.30 / 1_000_000
    )


def format_call_cost(usage: dict) -> str:
    cost = _compute_cost(usage)
    cached = "HIT" if usage["cache_read"] > 0 else ("WRITE" if usage["cache_creation"] > 0 else "MISS")
    return (
        f"${cost:.4f} | "
        f"in={usage['input']} out={usage['output']} "
        f"cache_read={usage['cache_read']} cache_write={usage['cache_creation']} "
        f"[{cached}]"
    )


def format_cost(usage: dict) -> str:
    cost = _compute_cost(usage)
    total_in = usage["input"] + usage["cache_read"] + usage["cache_creation"]
    cache_pct = 100 * usage["cache_read"] / total_in if total_in > 0 else 0
    return (
        f"~${cost:.3f} | "
        f"in={usage['input']:,} out={usage['output']:,} "
        f"cache_write={usage['cache_creation']:,} cache_read={usage['cache_read']:,} "
        f"(cache hit {cache_pct:.0f}%)"
    )


def run_phase(
    client: anthropic.Anthropic,
    phase_records: list[dict],
    done_ids: set[str],
    output_path: Path,
    delay: float,
    phase_name: str,
    usage_totals: dict,
) -> int:
    """Process records for a phase, skipping already-done IDs. Returns number of newly written records."""
    remaining = [r for r in phase_records if r["id"] not in done_ids]
    total = len(phase_records)
    already_done = total - len(remaining)

    print(f"\n=== {phase_name} ===")
    print(f"Target: {total} | Already done: {already_done} | Remaining: {len(remaining)}")

    if not remaining:
        print("Phase complete.")
        return 0

    written = 0
    with open(output_path, "a") as out_f:
        for i, record in enumerate(remaining):
            print(f"[{i+1}/{len(remaining)}] {record['id']}", end=" ... ", flush=True)
            result, call_usage = call_claude(client, record, usage_totals)
            if result is None:
                print("SKIP")
                continue
            out_f.write(json.dumps({
                "id": record["id"],
                "img_path": record["img_path"],
                "text": record["text"],
                "source": record["source"],
                "ground_truth_label": record["label"],
                **result,
            }) + "\n")
            out_f.flush()
            written += 1
            done_ids.add(record["id"])
            print(f"{result['classification']} ({result['probability']:.2f}) | {format_call_cost(call_usage)}")
            time.sleep(delay)

    print(f"  Usage so far: {format_cost(usage_totals)}")
    return written


def main():
    load_dotenv()  # Load environment variables from .env file if present
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/finetuning/dataset.jsonl")
    parser.add_argument("--hateful_target", type=int, default=1000,
                        help="Number of hateful memes to annotate (Phase 1)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--api_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_balance", type=float, default=3.0,
                        help="Minimum USD credit balance to proceed with Phase 2 (non-hateful)")
    parser.add_argument("--skip_non_hateful", action="store_true",
                        help="Skip Phase 2 (non-hateful memes) entirely")
    parser.add_argument("--force_non_hateful", action="store_true",
                        help="Run Phase 2 without checking credit balance")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Set ANTHROPIC_API_KEY or pass --api_key")

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_ids = load_eval_ids(data_dir)
    if eval_ids:
        print(f"Excluding {len(eval_ids)} eval records from sampling pool.")
    records = load_hateful_memes(data_dir, exclude_ids=eval_ids)
    n_hateful_avail = sum(r["label"] == 1 for r in records)
    n_non_hateful_avail = sum(r["label"] == 0 for r in records)
    print(f"Available: {len(records)} total ({n_hateful_avail} hateful, {n_non_hateful_avail} not_hateful)")

    # hateful memes
    hateful_sample = sample_phase(records, label=1, target=args.hateful_target, seed=args.seed)

    # non-hateful, to reach the 60/40 split
    # 60% hateful → total = hateful_target / 0.6, non-hateful = total * 0.4
    non_hateful_target = round(args.hateful_target * 2 / 3)  # 1000 hateful -> 667 non-hateful ≈ 60/40
    non_hateful_sample = sample_phase(records, label=0, target=non_hateful_target, seed=args.seed + 1)

    done_ids = load_done_ids(output_path)
    done_hateful, done_non_hateful = count_done_by_label(output_path)
    print(f"Already done: {done_hateful} hateful, {done_non_hateful} not_hateful")

    client = anthropic.Anthropic(api_key=args.api_key)
    usage_totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    # hateful
    run_phase(client, hateful_sample, done_ids, output_path, args.delay, "Phase 1 — Hateful memes", usage_totals)

    # Balance check before Phase 2 
    if args.skip_non_hateful:
        print("\nSkipping Phase 2 (--skip_non_hateful set).")
    else:
        if args.force_non_hateful:
            print("\nSkipping balance check (--force_non_hateful set). Starting Phase 2.")
            proceed = True
            balance_usd = None
        else:
            print(f"\nChecking credit balance (min required: ${args.min_balance:.2f})...")
            proceed, balance_usd = check_credit_balance(client, args.min_balance)
            if balance_usd is not None:
                print(f"  Balance: ${balance_usd:.2f} — {'sufficient' if proceed else 'INSUFFICIENT, skipping Phase 2'}")
            else:
                print("  Balance check unavailable — proceeding with Phase 2 by default.")
                print("  Pass --skip_non_hateful to abort, or --force_non_hateful to suppress this message.")

        if proceed:
            run_phase(client, non_hateful_sample, done_ids, output_path, args.delay, "Phase 2 — Non-hateful memes", usage_totals)
        else:
            print(f"Insufficient balance (${balance_usd:.2f} < ${args.min_balance:.2f}). Stopping after Phase 1.")

    done_hateful_final, done_non_hateful_final = count_done_by_label(output_path)
    total_final = done_hateful_final + done_non_hateful_final
    ratio = done_hateful_final / total_final if total_final > 0 else 0
    print(f"\nFinal dataset: {total_final} records — {done_hateful_final} hateful ({ratio:.1%}), {done_non_hateful_final} not_hateful ({1-ratio:.1%})")
    print(f"Total cost this run: {format_cost(usage_totals)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

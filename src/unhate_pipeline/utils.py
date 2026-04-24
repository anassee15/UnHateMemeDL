import re
import json


def parse_hateful_response(response):
    splitted_response = response.strip().replace("\n", " ").split("Classification:")
    description = splitted_response[0].strip()
    classification = splitted_response[1].split("Probability of the meme being hateful (from 0 to 1):")[0].strip().lower()
    probability_str = splitted_response[1].split("Probability of the meme being hateful (from 0 to 1):")[1].strip().rstrip("%").replace(",", ".")
    try:
        probability = float(probability_str)
    except ValueError:
        raise ValueError(f"Could not parse probability value: '{probability_str}'")

    is_hateful = classification == "hateful"
    return is_hateful, probability, description


def parse_hate_type_response(response):
    classification = response.replace("\n", " ").strip().split("Classification:")[1].strip().lower()
    return classification


def parse_hate_source_response(response):
    response = response.strip().lower()
    if "hate from image" in response:
        return "hate from image"
    elif "hate from text" in response:
        return "hate from text"
    elif "hate from both" in response:
        return "hate from both"
    else:
        raise ValueError("Response format is incorrect. Expected 'hate from image', 'hate from text', or 'hate from both'.")


def parse_prompt_generation(raw: str, fallback_prompt: str = "Preserve the image exactly as is.") -> dict:

    # 1. Strip <think> blocks and markdown fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    # 2. Extract outermost { } block with bracket counter
    start = raw.find("{")
    if start == -1:
        return _fallback(fallback_prompt, "No JSON block found")

    depth = 0
    in_string = False
    escaped = False

    for i, ch in enumerate(raw[start:], start):
        # Skip special handling for escaped characters inside strings.
        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = raw[start:i + 1]
                break
    else:
        return _fallback(fallback_prompt, "Unbalanced braces")

    # 3. Fix common LLM JSON issues
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str) # trailing commas
    json_str = re.sub(r'\bNone\b', 'null',  json_str) # Python None
    json_str = re.sub(r'\bTrue\b', 'true',  json_str) # Python True
    json_str = re.sub(r'\bFalse\b', 'false', json_str) # Python False
    json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)  # control chars

    # 4. Parse with fallback to ast.literal_eval
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            import ast
            parsed = ast.literal_eval(json_str)
        except Exception as e:
            return _fallback(fallback_prompt, f"Parse error: {e}")

    # 5. Validate flux_prompt is a clean plain string
    flux = parsed.get("flux_prompt", "")
    if not isinstance(flux, str) or len(flux.strip()) < 10 or flux.strip().startswith("{"):
        return _fallback(fallback_prompt, f"Invalid flux_prompt: '{flux[:80]}'")

    return parsed


def _fallback(prompt: str, reason: str) -> dict:
    print(f"[WARN] VLM parse failed: {reason}")
    return {
        "hate_source": "parse_error", "hate_location": "VISUAL_ONLY",
        "severity": "STRUCTURAL", "original_text": None,
        "replacement_text": None, "strategy": "Fallback: no mitigation applied",
        "flux_prompt": prompt, "expected_change": "Image unchanged due to parse error",
        "_parse_error": reason,
    }


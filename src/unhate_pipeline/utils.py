import re
import json


def parse_hateful_response(response):
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    raw = response.strip()
    if not raw:
        raise ValueError("Empty hateful detection response")

    # Support optional markdown code fences around JSON.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in hateful detection response: {e}")

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")

    description = str(parsed.get("description", "")).strip()
    classification = str(parsed.get("classification", "")).strip().lower()
    probability_raw = parsed.get("probability")

    if classification not in {"hateful", "non-hateful"}:
        raise ValueError(f"Invalid classification value: '{classification}'")

    is_hateful = classification == "hateful"

    if probability_raw is None:
        # Fine-tuned models trained without probability in the target schema will
        # omit this field. Fall back to the hard binary value from classification.
        probability = 1.0 if is_hateful else 0.0
    else:
        try:
            probability = float(probability_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Could not parse probability value: '{probability_raw}'")

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

    # 5. Validate required fields.
    diff = parsed.get("diffusion_prompt")
    if not isinstance(diff, str) or len(diff.strip()) < 10 or diff.strip().startswith("{"):
        snippet = "" if not isinstance(diff, str) else diff[:80]
        return _fallback(fallback_prompt, f"Invalid diffusion_prompt: '{snippet}'")

    _VALID_LOCATIONS = {"VISUAL_ONLY", "TEXT_ONLY", "COMBINED", "INTERSECTIONAL"}
    hate_loc = parsed.get("hate_location")
    if hate_loc not in _VALID_LOCATIONS:
        return _fallback(fallback_prompt, f"Missing or invalid hate_location: '{hate_loc}'")

    return parsed


def _fallback(prompt: str, reason: str) -> dict:
    print(f"[WARN] VLM parse failed: {reason}")
    return {
        "hate_source": "parse_error",
        "hate_location": "VISUAL_ONLY",
        "diffusion_prompt": prompt,
        "original_text": None,
        "replacement_text": None,
        "_parse_error": reason,
    }

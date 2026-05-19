from prompt import HATEFUL_DEFINITION


AFFECT_CLASSIFICATION_PROMPT = (
    "You are analyzing a meme or multimodal image before hatefulness detection.\n\n"
    "Your task is to classify the affective and communicative properties of the meme. "
    "Do NOT decide whether the meme is hateful in this step.\n\n"

    "Use the following meme affect categories inspired by Memotion Analysis / SemEval-2020 Task 8:\n"
    "- overall_sentiment: positive, neutral, negative\n"
    "- humor: not_funny, funny, very_funny, hilarious\n"
    "- sarcasm: not_sarcastic, little_sarcastic, very_sarcastic, extremely_sarcastic\n"
    "- offense: not_offensive, slight_offensive, very_offensive, hateful_offensive\n"
    "- motivation: not_motivational, motivational\n\n"

    "Definitions of the affect categories:\n\n"

    "Overall sentiment refers to the general affective polarity of the meme: positive, neutral, or negative. "
    "This label describes the general emotional tone of the meme, not whether it is hateful.\n\n"

    "Humor is understood here as a communicative mode in which a perceived violation, incongruity, "
    "or norm-breaking element is framed as benign, acceptable, or funny. Humor alone does not imply "
    "hatefulness or non-hatefulness.\n\n"

    "Sarcasm is understood here as a communicative mode in which the intended meaning may differ from, "
    "or even oppose, the literal meaning, often to mock, criticize, insult, irritate, or amuse. "
    "Sarcasm is context-dependent and should not be interpreted only literally.\n\n"

    "Offensive content is understood here as content that may be rude, insulting, profane, vulgar, abusive, "
    "or socially inappropriate. Offensive content is not automatically hate speech.\n\n"

    "Motivational content is understood here as content that aims to encourage, inspire, praise, or promote "
    "a desired attitude, identity, or behavior. Motivational framing is not hateful by itself.\n\n"

    "Important rules:\n"
    "- Classify only the meme's affective style, not its hatefulness.\n"
    "- Offensive does not automatically mean hateful.\n"
    "- Sarcastic does not automatically mean hateful.\n"
    "- Funny does not automatically mean non-hateful.\n"
    "- Sentiment polarity does not automatically determine hatefulness.\n"
    "- If uncertain between two intensity levels, choose the lower-intensity label.\n\n"

    "Return ONLY valid JSON with no extra text, markdown, or code fences.\n\n"
    "Required JSON schema:\n"
    "{\n"
    "  \"overall_sentiment\": \"positive | neutral | negative\",\n"
    "  \"humor\": \"not_funny | funny | very_funny | hilarious\",\n"
    "  \"sarcasm\": \"not_sarcastic | little_sarcastic | very_sarcastic | extremely_sarcastic\",\n"
    "  \"offense\": \"not_offensive | slight_offensive | very_offensive | hateful_offensive\",\n"
    "  \"motivation\": \"not_motivational | motivational\",\n"
    "  \"rationale\": \"<short explanation of the affective classification>\"\n"
    "}"
)


AFFECT_DEFINITIONS = {
    "overall_sentiment": (
        "Overall sentiment refers to the general affective polarity of the meme: positive, neutral, or negative. "
        "Sentiment polarity alone does not determine hatefulness. A negative meme is not automatically hateful, "
        "and a positive or neutral meme can still be hateful if it attacks, dehumanizes, threatens, excludes, "
        "humiliates, or endorses harm against a protected group."
    ),

    "humor": (
        "Humor is understood here as a communicative mode in which a perceived violation, incongruity, "
        "or norm-breaking element is framed as benign, acceptable, or funny. Humor alone is not hatefulness. "
        "The key question is whether the joke remains benign or whether it depends on attacking, humiliating, "
        "dehumanizing, excluding, or legitimizing harm against people based on protected characteristics."
    ),

    "sarcasm": (
        "Sarcasm is understood here as a communicative mode in which the intended meaning may differ from, "
        "or even oppose, the literal meaning, often to mock, criticize, insult, irritate, or amuse. "
        "When assessing hatefulness in sarcasm, do not rely only on the literal meaning. Assess the implied meaning, "
        "the implied target, and whether the sarcastic meaning attacks, dehumanizes, threatens, excludes, humiliates, "
        "or endorses harm against a protected group."
    ),

    "offense": (
        "Offensive content is understood here as content that may be rude, insulting, profane, vulgar, abusive, "
        "or socially inappropriate. Offensive content is not automatically hate speech. Distinguish general "
        "offensiveness from hatefulness. Classify as hateful only if the offensive content targets people based "
        "on protected characteristics or promotes violence, dehumanization, inferiority, exclusion, or segregation "
        "against such groups."
    ),

    "motivation": (
        "Motivational content is understood here as content that aims to encourage, inspire, praise, or promote "
        "a desired attitude, identity, or behavior. Motivational framing is not hateful by itself. Assess whether "
        "the motivational message promotes superiority, exclusion, contempt, or harm toward people based on "
        "protected characteristics."
    ),
}


def build_affect_aware_hateful_detection_prompt(affect: dict) -> str:
    overall_sentiment = affect.get("overall_sentiment", "neutral")
    humor = affect.get("humor", "not_funny")
    sarcasm = affect.get("sarcasm", "not_sarcastic")
    offense = affect.get("offense", "not_offensive")
    motivation = affect.get("motivation", "not_motivational")
    rationale = affect.get("rationale", "")

    detected_labels = (
        "Detected affective labels:\n"
        f"- overall_sentiment: {overall_sentiment}\n"
        f"- humor: {humor}\n"
        f"- sarcasm: {sarcasm}\n"
        f"- offense: {offense}\n"
        f"- motivation: {motivation}\n"
        f"- affective rationale: {rationale}\n"
    )

    affect_definitions = [
        AFFECT_DEFINITIONS["overall_sentiment"],
    ]

    interpretation_rules = []

    interpretation_rules.append(
        "Apply the general hatefulness definition consistently. The affective labels are not themselves evidence "
        "of hatefulness; they only guide interpretation."
    )

    if humor != "not_funny":
        affect_definitions.append(AFFECT_DEFINITIONS["humor"])
        interpretation_rules.append(
            f"The meme was classified as humorous with label '{humor}'. "
            "Do not classify it as hateful only because it is humorous. "
            "Assess whether the joke depends on attacking, humiliating, dehumanizing, excluding, "
            "or legitimizing harm against a protected group. If the humor does not target a protected group, "
            "do not classify it as hateful."
        )

    if sarcasm != "not_sarcastic":
        affect_definitions.append(AFFECT_DEFINITIONS["sarcasm"])
        interpretation_rules.append(
            f"The meme was classified as sarcastic with label '{sarcasm}'. "
            "Do not rely only on the literal meaning. Assess the implied target and implied meaning. "
            "Classify it as hateful only if the implied sarcastic meaning attacks, dehumanizes, threatens, excludes, "
            "humiliates, or endorses harm against a protected group."
        )

    if offense != "not_offensive":
        affect_definitions.append(AFFECT_DEFINITIONS["offense"])
        interpretation_rules.append(
            f"The meme was classified as offensive with label '{offense}'. "
            "Offensive content is not automatically hate speech. Distinguish general offensiveness from hatefulness. "
            "Classify it as hateful only if the offensive content targets protected characteristics or promotes "
            "violence, dehumanization, inferiority, exclusion, or segregation against protected groups."
        )

    if motivation != "not_motivational":
        affect_definitions.append(AFFECT_DEFINITIONS["motivation"])
        interpretation_rules.append(
            "The meme was classified as motivational. Motivational framing is not hateful by itself. "
            "Assess whether the motivational message promotes superiority, exclusion, contempt, or harm toward "
            "a protected group. Otherwise, do not classify it as hateful."
        )

    affect_definitions_text = "\n\n".join(dict.fromkeys(affect_definitions))
    interpretation_rules_text = "\n\n".join(interpretation_rules)

    return (
        "GENERAL HATEFULNESS DEFINITION:\n"
        + HATEFUL_DEFINITION + "\n\n"

        "AFFECTIVE CONTEXT:\n"
        + detected_labels + "\n"

        "DEFINITIONS OF RELEVANT AFFECTIVE CATEGORIES:\n"
        + affect_definitions_text + "\n\n"

        "AFFECT-AWARE INTERPRETATION RULES:\n"
        + interpretation_rules_text + "\n\n"

        "CLASSIFICATION CRITERIA:\n"
        "A meme is considered hateful if either the image or text is hateful on its own. "
        "The meme is also hateful if both the image and text are not hateful individually, "
        "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and "
        "they do not together create a hateful meaning, the meme is not considered hateful.\n\n"

        "FINAL DECISION RULE:\n"
        "Classify the meme as hateful only if the image, the text, or their combination attacks, dehumanizes, "
        "threatens, excludes, humiliates, or endorses harm against people based on protected characteristics. "
        "Do not classify a meme as hateful only because it is sarcastic, humorous, offensive, negative, shocking, "
        "or motivational.\n\n"

        "Return ONLY valid JSON with no extra text, markdown, or code fences.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"description\": \"<short explanation focused on hateful elements>\",\n"
        "  \"classification\": \"hateful | non-hateful\",\n"
        "  \"probability\": <number from 0 to 1>\n"
        "}\n\n"
        "Rules: description must be concise, classification must be exactly 'hateful' or 'non-hateful', "
        "and probability must be a numeric value in [0, 1]."
    )
from prompt import HATEFUL_DEFINITION


MEME_CATEGORY_PROMPT = (
    "Classify this meme into exactly one of the following three categories:\n\n"

    "1. historical — The meme references specific historical events, figures, wars, genocides, atrocities, "
    "or political movements (e.g., World War II, the Holocaust, colonialism, the Civil Rights movement, "
    "the Jim Crow era, the Rwandan genocide, slavery, the Armenian genocide, apartheid, Nazism, "
    "historical antisemitism, historical racism, or similar documented historical topics).\n\n"

    "2. general_culture — The meme is about everyday life, pop culture, entertainment, sports, food, "
    "internet culture, meme templates, relatable situations, work, relationships, or general humor "
    "not primarily tied to specific social identity groups or historical events.\n\n"

    "3. identity_social — The meme primarily targets, references, or concerns people based on a "
    "social identity group: race, religion, gender, nationality, ethnicity, sexual orientation, "
    "disability, or immigration status — in a contemporary (non-historical) context.\n\n"

    "Classification rules:\n"
    "- If the meme uses historical imagery or language to attack a contemporary group, choose 'historical'.\n"
    "- If the meme is ambiguous between general_culture and identity_social, choose identity_social.\n"
    "- Choose the single most applicable category; do not combine.\n\n"

    "Return ONLY valid JSON with no extra text, markdown, or code fences.\n"
    "Required JSON schema:\n"
    "{\n"
    "  \"category\": \"historical | general_culture | identity_social\",\n"
    "  \"rationale\": \"<one-sentence explanation of the classification>\"\n"
    "}"
)


CATEGORY_AFFECT_RULES = {
    "historical": (
        "This meme has been classified as a HISTORICAL meme — it references historical events, "
        "figures, atrocities, or political movements.\n\n"
        "For historical memes, pay special attention to:\n"
        "- Sarcasm that trivializes genocide, slavery, colonialism, or other historical atrocities.\n"
        "- Humor that uses the suffering of historical victims as a punchline.\n"
        "- 'Motivational' framing that promotes historical supremacist or genocidal ideologies.\n"
        "- Negative or positive sentiment directed at groups historically victimized (e.g., "
        "celebratory sentiment toward a genocidal figure is a strong hate signal).\n"
        "- Offense levels must be interpreted in their historical context: content that appears "
        "merely offensive may be hateful when its historical target group is considered.\n"
        "Historical framing does not make hateful content non-hateful. If the meme glorifies, "
        "trivializes, or re-promotes historical hatred, classify it as hateful."
    ),

    "general_culture": (
        "This meme has been classified as a GENERAL CULTURE meme — it is about everyday life, "
        "pop culture, entertainment, or general humor.\n\n"
        "For general culture memes, apply standard affect-aware analysis:\n"
        "- Humor and sarcasm are common and do not automatically indicate hatefulness.\n"
        "- Offensive content in everyday contexts (profanity, crude humor) differs from hate speech.\n"
        "- Check whether humor or sarcasm targets a specific protected group; if not, lean non-hateful.\n"
        "- Motivational content is usually benign unless it promotes group superiority.\n"
        "Use the affective labels to understand the communicative style, then assess whether any "
        "protected group is attacked, dehumanized, or targeted."
    ),

    "identity_social": (
        "This meme has been classified as an IDENTITY/SOCIAL meme — it primarily concerns people "
        "based on race, religion, gender, nationality, sexual orientation, disability, or ethnicity.\n\n"
        "For identity/social memes, apply heightened scrutiny:\n"
        "- Offensive content targeting protected groups is a strong indicator of hate speech.\n"
        "- Sarcasm or humor that mocks, stereotypes, or demeans an identity group is often hateful, "
        "even when framed as a joke.\n"
        "- Seemingly positive or neutral sentiment can mask dehumanizing stereotypes.\n"
        "- Motivational content that promotes one group's superiority over another is hateful.\n"
        "- Coded language, in-group references, and dog whistles are common in identity-targeted memes; "
        "assess implied meaning, not only literal meaning.\n"
        "The higher the offense level and the more specifically a protected group is targeted, "
        "the more likely the meme is hateful."
    ),
}


def build_category_aware_hateful_detection_prompt(category: str, affect: dict) -> str:
    category_rule = CATEGORY_AFFECT_RULES.get(
        category, CATEGORY_AFFECT_RULES["general_culture"]
    )

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

    return (
        "GENERAL HATEFULNESS DEFINITION:\n"
        + HATEFUL_DEFINITION + "\n\n"

        "MEME CATEGORY CONTEXT:\n"
        + category_rule + "\n\n"

        "AFFECTIVE CONTEXT:\n"
        + detected_labels + "\n"

        "CLASSIFICATION CRITERIA:\n"
        "A meme is considered hateful if either the image or text is hateful on its own. "
        "The meme is also hateful if both the image and text are not hateful individually, "
        "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and "
        "they do not together create a hateful meaning, the meme is not considered hateful.\n\n"

        "FINAL DECISION RULE:\n"
        "Classify the meme as hateful only if the image, the text, or their combination attacks, dehumanizes, "
        "threatens, excludes, humiliates, or endorses harm against people based on protected characteristics. "
        "Use the meme category context and affective labels to guide interpretation. "
        "Do not classify a meme as hateful only because it is sarcastic, humorous, offensive, "
        "negative, shocking, or motivational.\n\n"

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


HISTORICAL_FEW_SHOT_EXAMPLES = (
    "The following examples illustrate hateful historical memes. Study them to understand "
    "the hate patterns before classifying the new meme.\n\n"

    "--- EXAMPLE 1 (hateful) ---\n"
    "Meme description: An image showing Nazi SS or swastika iconography combined with text "
    "praising Adolf Hitler or stating that a particular ethnic or religious group (e.g., Jewish people) "
    "'controls' the media, banking, or government.\n"
    "Why it is hateful: This meme deploys Nazi-era antisemitic propaganda imagery alongside "
    "classic conspiracy theories used to dehumanize Jewish people. Even if framed as a joke or "
    "'historical fact', it legitimizes the ideology responsible for the systematic extermination "
    "of millions. The combination of iconography and stereotyping constitutes a direct attack on "
    "people based on religion and ethnicity.\n\n"

    "--- EXAMPLE 2 (hateful) ---\n"
    "Meme description: A meme using exaggerated racial caricatures (e.g., blackface imagery, "
    "monkey comparisons, or Jim Crow-era stereotypes about Black Americans linked to watermelons, "
    "fried chicken, or criminality).\n"
    "Why it is hateful: These caricatures originated as tools to dehumanize Black Americans and "
    "justify slavery and racial segregation. Even when wrapped in humor, they reinforce inferiority "
    "narratives, invoke a history of racial terror, and signal contempt for Black people as a group. "
    "The historical origin does not sanitize present-day use.\n\n"

    "--- EXAMPLE 3 (hateful) ---\n"
    "Meme description: A meme that uses the term 'cockroaches', 'vermin', or 'rats' to describe "
    "an ethnic or religious group, or that celebrates or jokes approvingly about a genocide "
    "(e.g., the Holocaust, the Rwandan genocide, the Armenian genocide).\n"
    "Why it is hateful: Dehumanizing animal metaphors for ethnic groups directly mirror the language "
    "used in documented genocides. Rwandan Hutu extremist radio called Tutsis 'inyenzi' (cockroaches) "
    "before and during the 1994 genocide. Even framed as dark humor or historical reference, "
    "content that approves of, trivializes, or celebrates mass killing based on ethnicity is hateful.\n\n"

    "--- EXAMPLE 4 (hateful) ---\n"
    "Meme description: A meme displaying Confederate iconography (the Confederate battle flag or "
    "statues of Confederate leaders) alongside text defending slavery, mocking the Civil Rights "
    "movement, or portraying Black Americans as undeserving of equal rights.\n"
    "Why it is hateful: The Confederate flag is a recognized symbol of white supremacy and racial "
    "oppression in the United States. Memes that deploy this symbol to demean Black Americans or "
    "dismiss civil rights invoke a documented history of racial terror, lynching, and segregation. "
    "Framing this as 'heritage' or historical commentary does not neutralize the attack.\n\n"

    "--- EXAMPLE 5 (hateful) ---\n"
    "Meme description: A reproduction of 1930s Nazi propaganda showing Jewish people as rats or "
    "parasites, captioned 'History doesn't lie' or 'Nothing has changed.'\n"
    "Why it is hateful: Recirculating genocide-era propaganda with implied contemporary approval "
    "directly dehumanizes Jewish people using the exact visual and rhetorical language that enabled "
    "the Holocaust. Framing it as 'historical fact' does not neutralize the attack — it amplifies it.\n\n"

    "--- EXAMPLE 6 (hateful) ---\n"
    "Meme description: Image from the Rwandan genocide or the Armenian genocide captioned with "
    "approval, celebration, or a joke (e.g., 'When you finally clean out the problem').\n"
    "Why it is hateful: Approving of or celebrating mass killing based on ethnicity is hateful "
    "regardless of framing. Humor does not neutralize genocide glorification.\n\n"

    "--- EXAMPLE 7 (not hateful, for contrast) ---\n"
    "Meme description: A photo of a historical figure (e.g., Martin Luther King Jr., Gandhi) "
    "with a famous inspirational quote, presented respectfully and without targeting any group.\n"
    "Why it is NOT hateful: The meme references history to inspire or inform. It does not attack, "
    "dehumanize, or demean any group.\n\n"
)


def build_historical_fewshot_detection_prompt() -> str:
    return (
        "GENERAL HATEFULNESS DEFINITION:\n"
        + HATEFUL_DEFINITION + "\n\n"

        "MEME CATEGORY CONTEXT:\n"
        "This meme has been classified as a HISTORICAL meme. Historical framing does not make "
        "hateful content non-hateful. Glorification, trivialization, or re-promotion of historical "
        "hatred is itself hate speech.\n\n"

        "FEW-SHOT EXAMPLES OF HATEFUL HISTORICAL MEMES:\n"
        + HISTORICAL_FEW_SHOT_EXAMPLES + "\n"

        "CLASSIFICATION CRITERIA:\n"
        "A meme is considered hateful if either the image or text is hateful on its own. "
        "The meme is also hateful if both the image and text are not hateful individually, "
        "but together convey a hateful message.\n\n"

        "FINAL DECISION RULE:\n"
        "Based on the hatefulness definition and the few-shot examples above, classify the new meme. "
        "Ask yourself: does this meme use historical content to attack, dehumanize, exclude, or "
        "promote harm against a group — in the same way as the hateful examples above?\n\n"

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

    "Definitions:\n\n"

    "Overall sentiment: the general affective polarity (positive, neutral, or negative). "
    "This is the emotional tone of the meme, not whether it is hateful.\n\n"

    "Humor: a communicative mode in which a perceived violation or incongruity is framed as funny. "
    "Humor alone does not imply hatefulness or non-hatefulness.\n\n"

    "Sarcasm: a communicative mode where the intended meaning may differ from or oppose the literal meaning. "
    "Sarcasm is context-dependent; do not interpret it only literally.\n\n"

    "Offensive content: content that may be rude, insulting, profane, vulgar, or socially inappropriate. "
    "Offensive content is not automatically hate speech.\n\n"

    "Motivational content: content that aims to encourage, inspire, or promote a desired attitude. "
    "Motivational framing is not hateful by itself.\n\n"

    "Rules:\n"
    "- Classify only the meme's affective style, not its hatefulness.\n"
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
        "or even oppose, the literal meaning. When assessing hatefulness, do not rely only on the literal "
        "meaning. Assess the implied meaning, the implied target, and whether the sarcastic message attacks, "
        "dehumanizes, threatens, excludes, humiliates, or endorses harm against a protected group."
    ),
    "offense": (
        "Offensive content is understood here as content that may be rude, insulting, profane, vulgar, abusive, "
        "or socially inappropriate. Offensive content is not automatically hate speech. Distinguish general "
        "offensiveness from hatefulness. Classify as hateful only if the offensive content targets people based "
        "on protected characteristics or promotes violence, dehumanization, inferiority, exclusion, or segregation."
    ),
    "motivation": (
        "Motivational content is understood here as content that aims to encourage, inspire, praise, or promote "
        "a desired attitude, identity, or behavior. Motivational framing is not hateful by itself. Assess whether "
        "the motivational message promotes superiority, exclusion, contempt, or harm toward people based on "
        "protected characteristics."
    ),
}


# Category-specific 2-example few-shot prompts. No affect step — just category
# routing then hate detection with short targeted examples.
# Example IDs per category — exclude all from evaluation metrics.

CATEGORIZED_V2_EXAMPLE_IDS = {
    "historical":      [9467, 80243],
    "general_culture": [84756, 86170],
    "identity_social": [62375, 69815],
}

_CV2_EXAMPLES = {
    "historical": (
        "FEW-SHOT EXAMPLES (historical hate):\n\n"
        "[09467.png] \"when mom asks where all the jews went\"\n"
        "→ Hateful: trivializes the Holocaust as a joke; mocking genocide is a direct attack on Jewish people based on religion and ethnicity.\n\n"
        "[80243.png] \"mississippi wind chime\"\n"
        "→ Hateful: coded phrase for a lynched Black person; uses racial murder as humor.\n\n"
    ),
    "general_culture": (
        "FEW-SHOT EXAMPLES (general culture hate):\n\n"
        "[84756.png] \"when you ask the kid that misses class a lot why he is bald and he starts crying\"\n"
        "→ Hateful: mocks a child's serious illness (implied cancer); targeting someone based on disease is hate speech.\n\n"
        "[86170.png] \"how to keep the thefts out\"\n"
        "→ Hateful: implies a racial group is inherently criminal; racial stereotype disguised as everyday humor.\n\n"
    ),
    "identity_social": (
        "FEW-SHOT EXAMPLES (identity/social hate):\n\n"
        "[62375.png] \"im not saying i'm racist im just saying all muslims are terrorists\"\n"
        "→ Hateful: equates an entire religious group with terrorism; direct stereotype-based attack on religion.\n\n"
        "[69815.png] \"thank you for allowing me into your country now we need to talk about things that need to change to accommodate my beliefs\"\n"
        "→ Hateful: caricatures immigrants as demanding outsiders; xenophobic attack based on national origin and religion.\n\n"
    ),
}

_CV2_CLASSIFICATION_CRITERIA = (
    "Classification criteria: A meme is considered hateful if either the image or text is hateful on its own. "
    "The meme is also hateful if both the image and text are not hateful individually, "
    "but together convey a hateful message."
)

_CV2_JSON_SCHEMA = (
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


def build_categorized_v2_detection_prompt(category: str) -> str:
    examples = _CV2_EXAMPLES.get(category, _CV2_EXAMPLES["general_culture"])
    return (
        HATEFUL_DEFINITION + "\n\n"
        + _CV2_CLASSIFICATION_CRITERIA + "\n\n"
        + examples
        + "Classify the meme based on the definition, criteria, and examples above. "
        + _CV2_JSON_SCHEMA
    )


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

    affect_definitions = [AFFECT_DEFINITIONS["overall_sentiment"]]
    interpretation_rules = [
        "Apply the general hatefulness definition consistently. The affective labels are not themselves evidence "
        "of hatefulness; they only guide interpretation."
    ]

    if humor != "not_funny":
        affect_definitions.append(AFFECT_DEFINITIONS["humor"])
        interpretation_rules.append(
            f"The meme was classified as humorous ('{humor}'). Do not classify it as hateful only because it "
            "is humorous. Assess whether the joke depends on attacking, humiliating, dehumanizing, or excluding "
            "a protected group. If the humor does not target a protected group, do not classify it as hateful."
        )

    if sarcasm != "not_sarcastic":
        affect_definitions.append(AFFECT_DEFINITIONS["sarcasm"])
        interpretation_rules.append(
            f"The meme was classified as sarcastic ('{sarcasm}'). Do not rely only on the literal meaning. "
            "Assess the implied target and implied meaning. Classify as hateful only if the implied sarcastic "
            "message attacks, dehumanizes, threatens, excludes, or endorses harm against a protected group."
        )

    if offense != "not_offensive":
        affect_definitions.append(AFFECT_DEFINITIONS["offense"])
        interpretation_rules.append(
            f"The meme was classified as offensive ('{offense}'). Offensive content is not automatically hate "
            "speech. Classify as hateful only if the offensive content targets protected characteristics or "
            "promotes violence, dehumanization, inferiority, exclusion, or segregation against protected groups."
        )

    if motivation != "not_motivational":
        affect_definitions.append(AFFECT_DEFINITIONS["motivation"])
        interpretation_rules.append(
            "The meme was classified as motivational. Motivational framing is not hateful by itself. Assess "
            "whether the motivational message promotes superiority, exclusion, contempt, or harm toward a "
            "protected group. Otherwise, do not classify it as hateful."
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

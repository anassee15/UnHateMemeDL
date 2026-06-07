# VLM prompt
# HATEFUL_DEFINITION must be defined first — affect_prompting.py imports it from
# this partially-initialized module during the circular import chain below.
HATEFUL_DEFINITION = (
    "Hatefulness definition: Hate speech is defined as a direct or indirect attack on people based "
    "on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, "
    "gender identity, sexual orientation, disability, disease, mocking hate crime or evoking pain from "
    "historical figures. The definition of attack includes violent or dehumanizing speech, statements of "
    "inferiority, and calls for exclusion or segregation."
)

# affect_prompting.py imports HATEFUL_DEFINITION from this module; because
# HATEFUL_DEFINITION is already defined above, the circular import resolves safely.
# (Always import `prompt` before `affect_prompting` standalone to keep this order.)
from affect_prompting import AFFECT_DEFINITIONS


HATEFUL_DETECTION_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    "Classification criteria: A meme is considered hateful if either the image or text is hateful on its own. "
    "The meme is also hateful if both the image and text are not hateful individually, "
    "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and "
    "they do not together create a hateful meaning, the meme is not considered hateful.\n\n"
    "You consider visual and textual meanings to classify the meme as hateful or non-hateful based on "
    "the hatefulness definition and classification criteria. Return ONLY valid JSON with no extra text, markdown, or code fences.\n\n"
    "Required JSON schema:\n"
    "{\n"
    "  \"description\": \"<short explanation focused on hateful elements>\",\n"
    "  \"classification\": \"hateful | non-hateful\",\n"
    "  \"probability\": <number from 0 to 1>\n"
    "}\n\n"
    "Rules: description must be concise, classification must be exactly 'hateful' or 'non-hateful', and probability must be a numeric value in [0, 1]."
)

# Simplified prompt for SFT (used by train_cls_head.py for embeddings and for
# the binary-only detection adapter). Only asks for the classification label.
# Probability is excluded — training targets are always exactly 0 or 1 (the
# binary ground truth), which would teach a degenerate confidence distribution.
HATEFUL_DETECTION_PROMPT_FT = (
    HATEFUL_DEFINITION + "\n\n"
    "Classification criteria: A meme is considered hateful if either the image or text is hateful on its own. "
    "The meme is also hateful if both the image and text are not hateful individually, "
    "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and "
    "they do not together create a hateful meaning, the meme is not considered hateful.\n\n"
    "You consider visual and textual meanings to classify the meme as hateful or non-hateful based on "
    "the hatefulness definition and classification criteria. Return ONLY valid JSON with no extra text, markdown, or code fences.\n\n"
    "Required JSON schema:\n"
    "{\n"
    "  \"classification\": \"hateful | non-hateful\"\n"
    "}\n\n"
    "Rules: classification must be exactly 'hateful' or 'non-hateful'."
)


# Rich SFT prompt for detection fine-tuning.
# Schema matches the training targets: classification + description + probability.
# probability is always 0.0 or 1.0 in training (binary ground truth) but the
# model learns to emit a calibrated float at inference.
HATEFUL_DETECTION_PROMPT_FT_RICH = (
    HATEFUL_DEFINITION + "\n\n"
    "Classification criteria: A meme is considered hateful if either the image or text is hateful on its own. "
    "The meme is also hateful if both the image and text are not hateful individually, "
    "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and "
    "they do not together create a hateful meaning, the meme is not considered hateful.\n\n"
    "Analyse the meme and return ONLY valid JSON — no extra text, markdown, or code fences.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "classification": "hateful | non-hateful",\n'
    '  "description": "<one sentence explaining precisely what makes this meme hateful, or why it is not hateful>",\n'
    '  "probability": <number from 0 to 1>\n'
    "}\n\n"
    "Rules: classification must be exactly 'hateful' or 'non-hateful'. "
    "description must be one concise sentence. "
    "probability must be a numeric value in [0, 1] reflecting confidence that the meme is hateful:\n"
    "- < 0.3 = clearly benign\n"
    "- > 0.7 = clearly hateful\n"
    "- 0.3-0.6 = ambiguous\n"
)


# Prompts for the --pipeline detection experiments in run_detection_eval.py.
# Additive: main's HATEFUL_DETECTION_PROMPT above is left untouched.

_JSON_SCHEMA = (
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

_CLASSIFICATION_CRITERIA = (
    "Classification criteria: A meme is considered hateful if either the image or text is hateful on its own. "
    "The meme is also hateful if both the image and text are not hateful individually, "
    "but together convey a hateful message. Conversely, if neither the image nor the text is hateful and "
    "they do not together create a hateful meaning, the meme is not considered hateful."
)


# zeroshot: definition + criteria only, no examples (matches main's baseline)
ZEROSHOT_DETECTION_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    + _CLASSIFICATION_CRITERIA + "\n\n"
    "You consider visual and textual meanings to classify the meme as hateful or non-hateful. "
    + _JSON_SCHEMA
)


# 17 synthetic calibration examples to raise confidence on borderline hateful memes
_BASELINE_FEW_SHOT = (
    "CALIBRATION EXAMPLES — Study these before classifying. They illustrate the full range of "
    "hate types so you can correctly assign probabilities above 0.5 to truly hateful content.\n\n"

    "Example 1 (explicit hate — direct slur + exclusion call):\n"
    "Image shows a Black person's face with overlaid text using a racial slur and the phrase "
    "'shouldn't be allowed to vote.'\n"
    "→ This is hateful because it uses a direct racial slur to attack a person based on race, "
    "combined with a call for political exclusion of an entire racial group. Probability should be very high (0.95+).\n\n"

    "Example 2 (explicit hate — slur + incitement to violence):\n"
    "Photo of a mosque with text 'One less [anti-Muslim slur] den — burn them all.'\n"
    "→ This is hateful because it uses a slur targeting Muslims and calls for violence against a religious "
    "group's place of worship. Direct incitement always warrants maximum probability.\n\n"

    "Example 3 (explicit hate — call for extermination):\n"
    "Photo of a gay pride parade with text 'Round them all up and execute them.'\n"
    "→ This is hateful because it directly calls for violence and extermination of people based on "
    "sexual orientation. No ambiguity — probability should be at maximum.\n\n"

    "Example 4 (implicit hate — humor format normalizing sexism):\n"
    "Drake meme: Drake disapproving label 'Equal pay for women' / Drake approving label "
    "'Women belong in the kitchen.'\n"
    "→ This is hateful because the humor format normalizes sexism; the punchline depends on treating "
    "women as inferior and undeserving of equal rights — a statement of inferiority based on gender. "
    "The joke format does not make it non-hateful.\n\n"

    "Example 5 (implicit hate — conspiracy theory framed as opinion):\n"
    "'Change my mind' meme with text 'Jews run Hollywood because they own everything.'\n"
    "→ This is hateful because it frames the antisemitic conspiracy theory of Jewish financial and "
    "media control as a debatable claim; the implied message attacks Jewish people collectively based "
    "on religion and ethnicity, even without an explicit slur.\n\n"

    "Example 6 (implicit hate — pseudoscientific racial stereotype):\n"
    "Exaggerated caricature of a Black person with text 'Science confirms' pointing to stereotyped behaviors.\n"
    "→ This is hateful because the caricature reproduces racial stereotypes historically used to "
    "dehumanize Black Americans; the 'science confirms' framing presents racist beliefs as fact, "
    "reinforcing inferiority narratives even when wrapped in humor.\n\n"

    "Example 7 (implicit hate — visual causation implying criminality):\n"
    "Two-panel meme: left panel shows a crowd of immigrants labeled 'Arriving,' right panel shows "
    "a crime scene labeled 'Predictable.'\n"
    "→ This is hateful because the visual juxtaposition implies a causal link between immigration "
    "and crime; it attacks people based on immigration status and national origin through implication "
    "rather than explicit text. Implied hate is still hate.\n\n"

    "Example 8 (culturally-coded hate — appropriated symbol):\n"
    "Pepe the Frog wearing a Nazi SS uniform, captioned 'Just a joke bro.'\n"
    "→ This is hateful because Pepe in Nazi iconography is a recognized white supremacist hate symbol; "
    "the 'just a joke' framing is a standard deflection tactic; the Nazi uniform promotes genocidal "
    "ideology. Cultural coding requires looking past the surface.\n\n"

    "Example 9 (culturally-coded hate — numerical white supremacist code):\n"
    "Image overlaid with large text '1488' and an American flag.\n"
    "→ This is hateful because '14' references the white nationalist 'Fourteen Words' slogan and "
    "'88' is numerical code for 'Heil Hitler' (H = 8th letter); together they constitute a white "
    "supremacist manifesto. Requires cultural knowledge of far-right codes to decode.\n\n"

    "Example 10 (culturally-coded hate — internet antisemitic caricature):\n"
    "Wojak-style cartoon of a figure with exaggerated stereotypical features labeled 'The International "
    "Banker' with a Star of David.\n"
    "→ This is hateful because it deploys Nazi-era antisemitic caricature tradition encoded in internet "
    "meme aesthetics; the Star of David links the caricature to Jewish people; the banking conspiracy "
    "framing attacks Jewish people collectively. Requires recognizing both historical antisemitic visual "
    "tradition and modern internet culture.\n\n"

    "Example 11 (historical hate — recycled genocide propaganda):\n"
    "Reproduction of a 1930s Nazi propaganda cartoon showing Jewish people as rats, "
    "captioned 'History doesn't lie.'\n"
    "→ This is hateful because it recirculates genocide-era propaganda with implied contemporary "
    "approval; the rat metaphor directly dehumanizes Jewish people using the exact language that "
    "enabled the Holocaust. Historical framing does not sanitize present-day use.\n\n"

    "Example 12 (historical hate — Confederate symbols + racial slur):\n"
    "Image of Confederate battle flag with text 'The Confederacy was right — the South will rise again. "
    "[racial slur] deserved what they got.'\n"
    "→ This is hateful because it celebrates the pro-slavery Confederate cause with an explicit slur; "
    "it uses historical framing to promote current racial hatred and justify subjugation of Black Americans.\n\n"

    "Example 13 (historical hate — genocide celebrated as humor):\n"
    "Photo from the Rwandan genocide captioned 'When you finally clean out the cockroaches lol.'\n"
    "→ This is hateful because it celebrates ethnic mass murder as humor; 'cockroaches' mirrors the exact "
    "dehumanizing language used to incite the 1994 genocide against Tutsis; approval of genocide is "
    "always hateful regardless of framing.\n\n"

    "Example 14 (historical hate — colonialism celebrated as achievement):\n"
    "Native American in traditional dress with text 'Thanksgiving: the day we civilized the savages — "
    "greatest holiday ever.'\n"
    "→ This is hateful because it trivializes the genocide and cultural destruction of Indigenous peoples; "
    "'savages' is a historical dehumanizing slur; framing colonization and mass killing as civilizational "
    "achievement attacks Indigenous people based on ethnicity and national origin.\n\n"

    "Example 15 (intersectional hate — race + religion + gender):\n"
    "Photo of a Black Muslim woman in a hijab with text 'Triple threat: Black, Muslim, and female — "
    "deport them all.'\n"
    "→ This is hateful because it simultaneously attacks race, religion, and gender; 'deport them all' "
    "is a call for exclusion based on multiple protected characteristics simultaneously.\n\n"

    "Example 16 (intersectional hate — antisemitism + anti-Black racism + white nationalism):\n"
    "Image of a Jewish family with text 'Jews breeding with Blacks to pollute the white race — "
    "the Great Replacement is real.'\n"
    "→ This is hateful because it combines antisemitic conspiracy theory with anti-Black racism and "
    "white nationalist 'Great Replacement' ideology; it attacks two ethnic/religious groups while "
    "promoting white supremacist ideology.\n\n"

    "Example 17 (intersectional hate — gender identity + national origin + disease dehumanization):\n"
    "Meme targeting transgender women from Latin American countries with text '[slur] from the south "
    "spreading disease — send them back.'\n"
    "→ This is hateful because it attacks gender identity with a slur, combines anti-immigrant "
    "sentiment, uses disease dehumanization, and calls for exclusion; it intersects three protected "
    "characteristics: gender identity, national origin, and immigration status.\n\n"
)

FEWSHOT_SYNTHETIC_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    + _CLASSIFICATION_CRITERIA + "\n\n"
    + _BASELINE_FEW_SHOT + "\n"
    "You consider visual and textual meanings to classify the meme as hateful or non-hateful based on "
    "the hatefulness definition, classification criteria, and calibration examples above. "
    + _JSON_SCHEMA
)


# 4 real few-shot examples; these IDs must be excluded from evaluation metrics
BASELINE_V2_EXAMPLE_IDS = [80243, 9467, 62375, 91756]

_BASELINE_V2_FEW_SHOT = (
    "FEW-SHOT EXAMPLES:\n\n"

    "[80243.png] \"mississippi wind chime\"\n"
    "→ Hateful: coded phrase for a lynched Black person; invokes racial murder and terror.\n\n"

    "[09467.png] \"when mom asks where all the jews went\"\n"
    "→ Hateful: treats the Holocaust as a joke; trivializing genocide is a direct attack on Jewish people.\n\n"

    "[62375.png] \"im not saying i'm racist im just saying all muslims are terrorists\"\n"
    "→ Hateful: equates an entire religious group with terrorism; explicit stereotype-based attack on religion.\n\n"

    "[91756.png] \"abnormal vs normal\"\n"
    "→ Hateful: labels a social identity group as abnormal, implying inferiority based on protected characteristics.\n\n"
)

BASELINE_V2_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    + _CLASSIFICATION_CRITERIA + "\n\n"
    + _BASELINE_V2_FEW_SHOT
    + "Classify the meme based on the definition, criteria, and examples above. "
    + _JSON_SCHEMA
)


# single call where the model reasons over affect dimensions before classifying
SINGLE_AFFECT_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"

    "AFFECT-AWARE CLASSIFICATION TASK\n\n"

    "Before reaching your final classification, analyze the affective properties of this meme "
    "across these dimensions:\n"
    "- Overall sentiment: is the emotional tone positive, neutral, or negative?\n"
    "- Humor: is this meant to be funny? (not_funny / funny / very_funny / hilarious)\n"
    "- Sarcasm: does the literal meaning differ from the intended meaning? "
    "(not_sarcastic / little_sarcastic / very_sarcastic / extremely_sarcastic)\n"
    "- Offensiveness: is this rude, vulgar, or socially inappropriate? "
    "(not_offensive / slight_offensive / very_offensive / hateful_offensive)\n"
    "- Motivation: does it aim to inspire or promote an attitude? (not_motivational / motivational)\n\n"

    "INTERPRETATION RULES — apply these based on the affective properties you detect:\n\n"

    + AFFECT_DEFINITIONS["overall_sentiment"] + "\n\n"
    + AFFECT_DEFINITIONS["humor"] + "\n\n"
    + AFFECT_DEFINITIONS["sarcasm"] + "\n\n"
    + AFFECT_DEFINITIONS["offense"] + "\n\n"
    + AFFECT_DEFINITIONS["motivation"] + "\n\n"

    "AFFECT-SPECIFIC GUIDANCE:\n"
    "- On humor: humor alone is not evidence of hatefulness. Ask whether the punchline requires "
    "attacking, humiliating, dehumanizing, or excluding a protected group. If not, do not classify as hateful.\n"
    "- On sarcasm: look past the literal meaning. Identify the implied target and the implied message. "
    "Classify as hateful only if the implied sarcastic message attacks a protected group.\n"
    "- On offensiveness: distinguish general offensiveness (profanity, crude humor) from hatefulness. "
    "Classify as hateful only if the offensive content targets protected characteristics.\n"
    "- On motivation: motivational framing is not hateful unless it promotes superiority or harm "
    "toward a protected group.\n"
    "- On sentiment: negative sentiment is not sufficient. A negative meme is not automatically "
    "hateful; a positive meme can still be hateful if it attacks a protected group.\n\n"

    + _CLASSIFICATION_CRITERIA + "\n\n"

    "FINAL DECISION RULE:\n"
    "Classify as hateful ONLY if the image, the text, or their combination attacks, dehumanizes, "
    "threatens, excludes, humiliates, or endorses harm against people based on protected characteristics. "
    "Do not classify as hateful only because the meme is sarcastic, humorous, offensive, negative, "
    "shocking, or motivational.\n\n"

    + _JSON_SCHEMA
)


TYPE_OF_HATE_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    "Classification Criteria: A meme is classified as unimodal-hate if either the image or "
    "the text is individually hateful. Conversely, a meme is classified as multimodal-hate if neither the image nor the text is "
    "hateful when considered individually, but together they convey a hateful message.\n\n"
    "The provided meme is considered hateful. To classify it as unimodal-hate or multimodal-hate, "
    "you analyze the hate in each of image and text parts individually based on the provided hatefulness definition. "
    "Then you give the answer based on the classification criteria in the following format.\n"
    "Explanation:\n"
    "Classification:"
)

SOURCE_OF_HATE_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    "The provided meme is considered **hateful**. Your task is to analyze whether the source "
    "of hate inside the meme is from image or text or both. Please answer with "
    "'hate from image', 'hate from text', or 'hate from both' **only**"
)


# Fine-tuning prompt for mitigation generation. Single user turn (no system),
# schema-locked to the 5 fields the downstream pipeline actually consumes.
# `hate_source` first so the model’s own description acts as a CoT scaffold
# before it emits the structured mitigation fields.

DIFFUSION_PROMPT_HEAD = """## Role
You are a content moderator and diffusion-model prompt engineer.

The image provided is a hateful meme. Produce the best mitigation plan to remove the hateful content while preserving the meme’s original intent, structure, and humor as much as possible.

## Output
Return ONLY one valid JSON object with exactly this structure:

{
  "hate_source": "<one sentence explaining what makes this meme hateful>",
  "hate_location": "VISUAL_ONLY" | "TEXT_ONLY" | "COMBINED" | "INTERSECTIONAL",
  "diffusion_prompt": "<2-4 sentences, visual-only description of the mitigated scene; never mention text or overlays>",
  "original_text": "<exact meme text, or null>",
  "replacement_text": "<rewritten meme text removing hate while preserving the joke/point, or null; at most one \\n>",
}

"""

DIFFUSION_PROMPT_THINK = """## Step-by-step reasoning (think before outputting):

<think>
Reason privately and do not question whether the meme is hateful; assume hateful content is present.

- Identify the protected group being targeted.
- Determine whether the hateful meaning comes from the image, the text, or both.
- Pinpoint the hateful mechanism: stereotype, mockery, humiliation, dehumanization, exclusion, threat, or glorification/minimization of historical violence.
- Preserve the meme’s original meaning, joke structure, and tone as much as possible while removing the hateful targeting.
- Change only what is necessary: visual only, text only, or one of each depending on hate_location.
- Copy original_text exactly.
- Write hate_source as one precise sentence explaining what is hateful and why.
- Write diffusion_prompt as 2-4 sentences describing only the mitigated visual scene, never mentioning text or overlays.
- Write replacement_text only when the text itself carries hateful meaning; otherwise use null.

Output only the final JSON object.
</think>

"""

DIFFUSION_PROMPT_RULES = """## Rules

### hate_location

- VISUAL_ONLY: image hateful, text neutral
- TEXT_ONLY: text hateful, image benign
- COMBINED: hate emerges from image + text together
- INTERSECTIONAL: both image and text are independently hateful
- null if not_hateful

### description

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


### Mitigation

Preserve the original semantic content, intent, joke structure, and communicative function of the meme as much as possible, while removing hateful targeting or stereotypes.

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
- at most one newline to preserve top text / bottom text structure if present
- null if original text is already non-hateful

Return ONLY the JSON object, no markdown fences, no explanation."""


def build_diffusion_prompt(thinking: bool = True) -> str:
    """Assemble the FT mitigation prompt, optionally including the CoT reasoning block."""
    parts = [DIFFUSION_PROMPT_HEAD]
    if thinking:
        parts.append(DIFFUSION_PROMPT_THINK)
    parts.append(DIFFUSION_PROMPT_RULES)
    return "".join(parts)


GET_DIFFUSION_PROMPT = build_diffusion_prompt(thinking=True)


# Diffusion prompt

ERASE_TEXT_PROMPT = (
    "Remove all text overlays, captions, and watermarks from this image. "
    "Repaint every text region to seamlessly match the surrounding background "
    "texture, color, and lighting. Preserve all non-text visual elements exactly."
)
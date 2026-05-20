# VLM prompt
HATEFUL_DEFINITION = (
    "Hatefulness definition: Hate speech is defined as a direct or indirect attack on people based "
    "on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, "
    "gender identity, sexual orientation, disability, disease, mocking hate crime or evoking pain from "
    "historical figures. The definition of attack includes violent or dehumanizing speech, statements of "
    "inferiority, and calls for exclusion or segregation."
)


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
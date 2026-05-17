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

# Simplified prompt for SFT fine-tuning: only asks for the binary classification
# label. Probability is excluded because training targets are always exactly 0 or 1
# (the binary label), which would teach the model a degenerate distribution rather
# than a calibrated confidence score.
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


GET_DIFFUSION_SYSTEM_PROMPT = (
    "You are an expert image content moderator and prompt engineer specialized in diffusion models, specifically FLUX Klein, "
    "with deep expertise in meme culture and internet visual language.\n\n"
    "Your role is a two-stage pipeline:\n"
    "1. ANALYZE the input image (which may be a meme) and identify hateful content — visual, textual, or the combination of both\n"
    "2. OUTPUT a mitigation prompt for FLUX Klein img2img that surgically removes hateful content\n\n"
    "You must follow these strict rules:\n"
    "- Identify the EXACT source of hate: visual elements, text overlays, or the combination of image+text that creates hate "
    "(a neutral image + hateful caption = hate meme)\n"
    "- Understand meme structure: TOP TEXT / BOTTOM TEXT / image macro / exploitable templates / screenshot memes / deep-fried memes\n"
    "- Preserve ALL non-hateful elements: meme format, humor style, subject, composition, font style, visual template\n"
    "- When hate lives in the TEXT: produce a replacement text that preserves the joke structure/punchline but removes the hateful "
    "target (redirect the humor at a neutral or self-referential target)\n"
    "- When hate lives in the VISUAL: use precise visual editing instructions\n"
    "- When hate emerges from IMAGE+TEXT combination: address both simultaneously\n"
    "- Minimize semantic drift: a meme that was funny should remain funny if possible, just not at the expense of a group\n"
    "- If full neutralization requires destroying the joke entirely, produce a prompt that transforms it into a clearly benign alternative\n"
    "- Never refuse to produce a mitigation prompt — neutralization is always possible\n"
    "- Never reproduce, describe approvingly, or amplify the hateful content in your reasoning"
)

GET_DIFFUSION_USER_PROMPT = (
    "You are given an image — potentially a meme — that has been flagged as containing hateful content.\n\n"
    "## Your Task\n\n"
    "Analyze the image carefully and produce a diffusion model (FLUX Klein) img2img editing prompt that mitigates the hateful content, "
    "handling both visual and textual elements.\n\n"
    "## Step-by-step reasoning (think before outputting):\n\n"
    "<think>\n"
    "1. IDENTIFY THE MEME STRUCTURE (if applicable):\n"
    "   - Is this a classic image macro (top text / bottom text)?\n"
    "   - Is the text embedded in the image (burned-in) or is it a caption?\n"
    "   - Is it a screenshot meme, a deep-fried meme, a wojak/pepe variant, a political cartoon?\n"
    "   - What is the meme's original format and intended humor mechanism?\n\n"
    "2. IDENTIFY THE SOURCE OF HATE — be specific about whether it is:\n"
    "   - TEXT-ONLY: the image is neutral but the text overlay is hateful\n"
    "     (e.g., a Drake meme where the text targets a racial/religious group)\n"
    "   - VISUAL-ONLY: the text is absent/neutral but the image contains hate symbols,\n"
    "     dehumanizing caricatures, or hate group iconography\n"
    "   - COMBINED: the image+text pair creates hate that neither would alone\n"
    "     (e.g., a neutral image of a group + a dehumanizing caption)\n"
    "   - INTERSECTIONAL: both image and text are independently hateful\n\n"
    "3. CLASSIFY severity:\n"
    "   - SURGICAL_TEXT: only the text needs changing, image is fine\n"
    "   - SURGICAL_VISUAL: only a visual element needs changing, text is fine\n"
    "   - SURGICAL_BOTH: small targeted changes to both text and visual (give a replacement text different from the original in that case)\n"
    "   - STRUCTURAL: the entire concept must be transformed\n\n"
    "4. DETERMINE mitigation strategy:\n"
    "   - For TEXT: craft replacement text that preserves the joke format/punchline (represents the difference between top and bottom text localisation with a newline, maximum one new line)\n"
    "     but redirects the target to something neutral (e.g., a universal frustration,\n"
    "     a self-referential tech/internet joke, an absurdist alternative) but by keeping the meaning the same\n"
    "   - For VISUAL: replace/remove the hateful element with a neutral equivalent non-hatefull but without changing the overall composition \n"
    "   - For COMBINED: address text first (as it often drives the hate), then visual\n\n"
    "5. DRAFT the diffusion prompt using plain natural language — it must be ready to pass\n"
    "   directly to pipe(prompt=...) with no extra parsing:\n"
    "   - For text changes: 'Never speak about text change in this part, the diffusion model should only handle visual changes'\n"
    "   - For visual changes: 'Replace [hateful element] with [neutral equivalent],\n"
    "     preserve all other visual elements including composition, lighting, and colors.'\n"
    "   - Always anchor preserved elements explicitly in the prompt\n"
    "</think>\n\n"
    "## Output Format\n\n"
    "Respond ONLY with the following JSON — no extra commentary.\n\n"
    "CRITICAL: The 'diffusion_prompt' field must be a plain natural language string, "
    "ready to be passed DIRECTLY to a diffusion model as pipe(prompt=...). "
    "It must NOT contain JSON, brackets, field names, or structured syntax. "
    "It must read as a natural image editing instruction, like a human art director "
    "briefing an image editor.\n\n"
    "{\n"
    '  "hate_source": "<one sentence: what element is hateful and why>",\n'
    '  "hate_location": "TEXT_ONLY | VISUAL_ONLY | COMBINED | INTERSECTIONAL",\n'
    '  "severity": "SURGICAL_TEXT | SURGICAL_VISUAL | SURGICAL_BOTH | STRUCTURAL",\n'
    '  "original_text": "<verbatim text visible in the image, or null>",\n'
    '  "replacement_text": "<neutral replacement text preserving humor structure (represents the difference between top and bottom text with a newline, maximum one new line), or null>",\n'
    '  "strategy": "<one sentence: what changes and what is preserved>",\n'
    '  "diffusion_prompt": "<plain natural language diffusion prompt, never speak about text change in this part, the diffusion model should only handle visual changes>",\n'
    '  "expected_change": "<one sentence: what the output will look like vs. input>"\n'
    "}\n\n"
    "## Examples of valid diffusion_prompt values:\n\n"
    "SURGICAL_TEXT example:\n"
    "'Remove the top text and bottom text overlays completely. Repaint the text areas\n"
    "to match the background texture. Preserve the original meme template image,\n"
    "composition, lighting, and colors exactly.'\n\n"
    "SURGICAL_VISUAL example:\n"
    "'Replace the hate symbol on the character's armband with a plain red armband.\n"
    "Keep all other elements identical: clothing style, pose, background, lighting,\n"
    "facial expression, and overall composition.'\n\n"
    "SURGICAL_BOTH example:\n"
    "'Remove the bottom caption text and repaint the area to match the background.\n"
    "Replace the hate group logo on the banner with a generic smiley face icon.\n"
    "Preserve the meme template format, font style, image macro layout, and all\n"
    "other visual elements.'\n\n"
    "STRUCTURAL example:\n"
    "'Transform this image into a wholesome version of the same meme format.\n"
    "Keep the Impact font style, two-panel layout, and overall composition.\n"
    "Replace all figures with cartoon animals. Remove all text overlays and\n"
    "repaint those regions to match the background.'"
)


# Fine-tuning prompt for mitigation generation. Single user turn (no system),
# schema-locked to the 5 fields the downstream pipeline actually consumes.
# `hate_source` first so the model’s own description acts as a CoT scaffold
# before it emits the structured mitigation fields.

_DIFFUSION_PROMPT_FT_HEAD = """## Role
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

_DIFFUSION_PROMPT_FT_THINK = """## Step-by-step reasoning (think before outputting):

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

_DIFFUSION_PROMPT_FT_RULES = """## Rules

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
    parts = [_DIFFUSION_PROMPT_FT_HEAD]
    if thinking:
        parts.append(_DIFFUSION_PROMPT_FT_THINK)
    parts.append(_DIFFUSION_PROMPT_FT_RULES)
    return "".join(parts)


GET_DIFFUSION_PROMPT = build_diffusion_prompt(thinking=True)


# Diffusion prompt

ERASE_TEXT_PROMPT = (
    "Remove all text overlays, captions, and watermarks from this image. "
    "Repaint every text region to seamlessly match the surrounding background "
    "texture, color, and lighting. Preserve all non-text visual elements exactly."
)
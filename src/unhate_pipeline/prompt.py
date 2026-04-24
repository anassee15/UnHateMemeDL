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
    "5. DRAFT the FLUX prompt using plain natural language — it must be ready to pass\n"
    "   directly to pipe(prompt=...) with no extra parsing:\n"
    "   - For text changes: 'Never speak about text change in this part, the diffusion model should only handle visual changes'\n"
    "   - For visual changes: 'Replace [hateful element] with [neutral equivalent],\n"
    "     preserve all other visual elements including composition, lighting, and colors.'\n"
    "   - Always anchor preserved elements explicitly in the prompt\n"
    "</think>\n\n"
    "## Output Format\n\n"
    "Respond ONLY with the following JSON — no extra commentary.\n\n"
    "CRITICAL: The 'flux_prompt' field must be a plain natural language string, "
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
    '  "flux_prompt": "<plain natural language diffusion prompt, never speak about text change in this part, the diffusion model should only handle visual changes>",\n'
    '  "expected_change": "<one sentence: what the output will look like vs. input>"\n'
    "}\n\n"
    "## Examples of valid flux_prompt values:\n\n"
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


# Diffusion prompt

ERASE_TEXT_PROMPT = (
    "Remove all text overlays, captions, and watermarks from this image. "
    "Repaint every text region to seamlessly match the surrounding background "
    "texture, color, and lighting. Preserve all non-text visual elements exactly."
)
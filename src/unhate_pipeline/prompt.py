# VLM prompt

# HATEFUL_DEFINITION must be defined first — affect_prompting.py imports it
# from this partially-initialized module during the circular import chain.
HATEFUL_DEFINITION = (
    "Hatefulness definition: Hate speech is defined as a direct or indirect attack on people based "
    "on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, "
    "gender identity, sexual orientation, disability, disease, mocking hate crime or evoking pain from "
    "historical figures. The definition of attack includes violent or dehumanizing speech, statements of "
    "inferiority, and calls for exclusion or segregation."
)

# affect_prompting.py imports HATEFUL_DEFINITION from this module; because
# HATEFUL_DEFINITION is already defined above, the circular import resolves safely.
from affect_prompting import AFFECT_DEFINITIONS


# ---------------------------------------------------------------------------
# Shared JSON schema footer (identical across all prompts)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# PIPELINE: baseline
# Baseline prompt + 17 calibration few-shot examples.
# Goal: raise model confidence so p=0.5 correctly separates hateful/non-hateful.
# (Empirically, very hateful memes were scoring 0.2–0.5 without examples.)
# ---------------------------------------------------------------------------

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

HATEFUL_DETECTION_PROMPT = (
    HATEFUL_DEFINITION + "\n\n"
    + _CLASSIFICATION_CRITERIA + "\n\n"
    + _BASELINE_FEW_SHOT + "\n"
    "You consider visual and textual meanings to classify the meme as hateful or non-hateful based on "
    "the hatefulness definition, classification criteria, and calibration examples above. "
    + _JSON_SCHEMA
)


# ---------------------------------------------------------------------------
# PIPELINE: baseline_v2
# Compact version: 4 short real-dataset few-shot examples (under 2000 chars total).
# Replaces the 17-example baseline which is too long for Qwen3.6.
# Example IDs — MUST be excluded from evaluation metrics.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PIPELINE: single_affect
# One VLM call. The prompt instructs the model to analyze affective dimensions
# (sentiment, humor, sarcasm, offense, motivation) internally before classifying.
# No separate sentiment step — all reasoning happens in a single call.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Other prompts — unchanged
# ---------------------------------------------------------------------------

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

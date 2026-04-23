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
    "the hatefulness definition and classification criteria. After giving a **short explanation**, provide the "
    "answer with either hateful or non-hateful in the following format:\n"
    "Classification:\n"
    "Probability of the meme being hateful (from 0 to 1):"
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
    "The provided meme is considered hateful. Your task is to analyze whether the source "
    "of hate inside the meme is from image or text or both. Please answer with "
    "'hate from image', 'hate from text', or 'hate from both'."
)

TEXT_MITIGATION_PROMPT = (
    "The provided meme is considered hateful. Your task is to analyze its hatefulness and generate "
    "substitute text such that the new meme is not hateful, using the same meme image. Please ensure "
    "the generated text aligns with the content of the image and that the tone of the text is similar. "
    "Please only provide the new text that you want to generate without any additional text or explanation, you **have to change the text to make it non-hateful**."
)


def parse_hateful_response(response):
    classification = response.strip().replace("\n", " ").split("Classification:")[1].split("Probability of the meme being hateful")[0].strip().lower()
    probability_str = response.strip().replace("\n", " ").split("Probability of the meme being hateful (from 0 to 1):")[1].strip().rstrip("%").replace(",", ".")
    try:
        probability = float(probability_str)
    except ValueError:
        raise ValueError(f"Could not parse probability value: '{probability_str}'")

    is_hateful = classification == "hateful"
    return is_hateful, probability


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
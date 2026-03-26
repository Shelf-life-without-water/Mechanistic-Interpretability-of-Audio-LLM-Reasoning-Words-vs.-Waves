
SYSTEM_AUDIO = "You are a careful speech emotion analyst."
SYSTEM_TEXT = "You are a careful text-based emotion analyst."


def labels_str(labels: list[str]) -> str:
    return ", ".join(labels)


def build_audio_only_prompt(processor, labels: list[str]) -> str:
    instruction = (
        f"Listen to the utterance and classify the speaker's expressed emotion. "
        f"Return exactly one lowercase label from: {labels_str(labels)}."
    )
    conv = [
        {"role": "system", "content": SYSTEM_AUDIO},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "LOCAL_AUDIO"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def build_text_only_prompt(processor, transcription: str, labels: list[str]) -> str:
    instruction = (
        f'Transcript: "{transcription}"\n'
        f"Using only the lexical content above, classify the speaker's emotion. "
        f"Return exactly one lowercase label from: {labels_str(labels)}."
    )
    conv = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": instruction},
    ]
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def build_audio_text_prompt(processor, transcription: str, labels: list[str]) -> str:
    instruction = (
        f'Transcript: "{transcription}"\n'
        f"Using both the audio and the transcript, classify the speaker's expressed emotion. "
        f"Return exactly one lowercase label from: {labels_str(labels)}."
    )
    conv = [
        {"role": "system", "content": SYSTEM_AUDIO},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "LOCAL_AUDIO"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def build_audio_hint_prompt(processor, transcription: str, hint_label: str, labels: list[str]) -> str:
    instruction = (
        f'Transcript: "{transcription}"\n'
        f'A transcript-only classifier predicts "{hint_label}". '
        f"Classify the actual emotion expressed in the audio. "
        f"Return exactly one lowercase label from: {labels_str(labels)}."
    )
    conv = [
        {"role": "system", "content": SYSTEM_AUDIO},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "LOCAL_AUDIO"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

MAPPING_EXTRACTOR_SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes the difference of prompts."
)

def build_mapping_extractor_user_prompt(
    pairs: list,
    source_model: str,
    target_model: str,
) -> str:
    """Build the Mapping Extractor user prompt from the paper (Section E.1)."""
    m = len(pairs)
    sections = []
    for i, pair in enumerate(pairs, 1):
        sections.append(
            f"Source Prompt {{{i}}}: {pair.source_result.optimal_prompt}\n"
            f"Target Prompt {{{i}}}: {pair.target_result.optimal_prompt}\n"
            f"Dataset: {pair.task_info}"
        )

    return (
        f"Below are {m} examples of the source prompts and target prompts, "
        f"along with their dataset and information on the dataset.\n\n"
        + "\n\n".join(sections)
        + "\n\nPlease summarize the common prompt difference of the source "
        f"prompts to the target prompts, also considering the dataset and "
        f"information."
    )

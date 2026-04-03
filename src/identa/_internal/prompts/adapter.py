def build_adapter_prompt(
    original_prompt: str,
    transfer_summary: str,
    source_model: str,
    target_model: str,
    task_context: str | None = None,
) -> str:
    """Build the Adapter Model prompt from the paper (Section E.1)."""
    context_note = ""
    if task_context:
        context_note = (
            f"\nThese transfer effects were derived from standard datasets "
            f"and must now be adapted for {task_context}.\n"
        )

    return (
        f"Your task is to generate a new target prompt by applying the "
        f"specified transfer effects to the Original Prompt.\n"
        f"{context_note}"
        f"The new prompt should:\n"
        f"- Begin from the provided Original Prompt.\n"
        f"- Incorporate the transfer effects summary faithfully.\n"
        f"- Be adapted for the {target_model} model.\n"
        f"- Remain concise and preserve the original meaning.\n"
        f"- Improve suitability for eliciting high-quality responses from "
        f"{target_model}.\n\n"
        f"## ====== Original Prompt ======\n"
        f"{original_prompt}\n"
        f"## ====== End Original Prompt ======\n\n"
        f"## ====== Transfer Effects Summary ======\n"
        f"{transfer_summary}\n"
        f"## ====== End Transfer Effects Summary ======\n\n"
        f"**Task:**\n"
        f"Apply the transfer effects summary to the Original Prompt "
        f"optimized for {source_model} and produce an optimized prompt "
        f"for {target_model}.\n\n"
        f"Optimized Prompt:"
    )

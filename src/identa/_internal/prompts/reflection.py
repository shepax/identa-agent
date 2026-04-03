def build_reflection_prompt(
    parent_prompt: str,
    task_description: str,
    question: str,
    expected_answer: str,
    model_response: str,
    performance_score: float,
    behavioral_breakdown=None,
) -> str:
    """Build the reflective evolution query for MAP-RPE."""
    feedback_lines = [f"Performance Score: {performance_score:.2f}"]
    if behavioral_breakdown:
        feedback_lines.extend([
            f"Syntax Validity: {behavioral_breakdown.syntax_validity:.2f}",
            f"Entry Point Defined: {behavioral_breakdown.entry_point_defined:.2f}",
            f"Risk-Free Patterns: {behavioral_breakdown.risk_free_patterns:.2f}",
            f"No Undesirable Patterns: {behavioral_breakdown.no_undesirable:.2f}",
            f"Behavioral Score: {behavioral_breakdown.total:.2f}",
        ])

    return (
        f"## Current Prompt Template\n"
        f"{parent_prompt}\n\n"
        f"## Task Description\n"
        f"{task_description}\n\n"
        f"## Example Question\n"
        f"{question}\n\n"
        f"## Expected Answer (excerpt)\n"
        f"{expected_answer}\n\n"
        f"## Model Response (excerpt)\n"
        f"{model_response}\n\n"
        f"## Evaluation Feedback\n"
        + "\n".join(feedback_lines)
        + "\n\n## Instructions\n"
        f"Analyze the evaluation feedback and identify specific weaknesses "
        f"in the current prompt template. Then produce an improved version "
        f"that addresses these weaknesses while preserving what works well.\n\n"
        f"Focus on:\n"
        f"- Clarity of instructions\n"
        f"- Output format specifications\n"
        f"- Edge case handling\n"
        f"- Model-specific alignment cues\n\n"
        f"You must structure your response using XML tags as follows:\n"
        f"1. `<analysis>`: Briefly explain why the current prompt failed based on the feedback.\n"
        f"2. `<plan>`: Outline the structural changes you will make.\n"
        f"3. `<prompt>`: The complete, revised prompt template.\n\n"
        f"Do not include any other text outside these tags."
    )

from identa.calibration.types import PromptCandidate
from identa.providers.base import ModelProvider, CompletionRequest
from identa.tasks.schema import AlignmentTask, TaskInstance
from identa._internal.prompts.reflection import build_reflection_prompt

class ReflectiveEvolver:
    """Generate improved prompt candidates via reflective feedback."""

    def __init__(
        self,
        reflection_provider: ModelProvider,
        reflection_model_id: str,
        exploitation_ratio: float = 0.7,
        exploration_ratio: float = 0.2,
        elite_selection_ratio: float = 0.1,
    ):
        self.provider = reflection_provider
        self.model_id = reflection_model_id
        self.exploitation_ratio = exploitation_ratio
        self.exploration_ratio = exploration_ratio
        self.elite_ratio = elite_selection_ratio

    async def evolve(
        self,
        parent: PromptCandidate,
        instance: TaskInstance,
        task: AlignmentTask,
        evaluation_feedback: dict,
    ) -> str:
        reflection_prompt = build_reflection_prompt(
            parent_prompt=parent.prompt_text,
            task_description=task.description,
            question=instance.question,
            expected_answer=instance.answer[:300],
            model_response=evaluation_feedback.get("response", ""),
            performance_score=evaluation_feedback.get("performance", 0),
            behavioral_breakdown=evaluation_feedback.get("behavioral"),
        )

        import re
        response = await self.provider.complete(
            CompletionRequest(
                messages=[
                    {"role": "system", "content":
                        "You are a prompt engineering expert. Your task is to "
                        "improve the given prompt template based on evaluation "
                        "feedback. You must perform a step-by-step analysis inside "
                        "`<analysis>` and `<plan>` tags, and finally output your "
                        "improved prompt inside `<prompt>` tags."},
                    {"role": "user", "content": reflection_prompt},
                ],
                model=self.model_id,
                temperature=0.7,  # Higher temp for diversity in evolution
                max_tokens=4096,
            )
        )

        content = response.content.strip()
        match = re.search(r"<prompt>(.*?)</prompt>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

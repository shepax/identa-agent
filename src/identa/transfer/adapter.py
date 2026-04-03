from identa.providers.base import ModelProvider, CompletionRequest
from identa._internal.prompts.adapter import build_adapter_prompt

class PromptAdapter:
    """Apply transfer mapping to adapt a source prompt for the target model."""

    def __init__(self, provider: ModelProvider, model_id: str):
        self.provider = provider
        self.model_id = model_id

    async def adapt(
        self,
        source_prompt_text: str,
        transferable_knowledge: str,
        source_model: str,
        target_model: str,
        task_context: str | None = None,
    ) -> str:
        adapter_prompt = build_adapter_prompt(
            original_prompt=source_prompt_text,
            transfer_summary=transferable_knowledge,
            source_model=source_model,
            target_model=target_model,
            task_context=task_context,
        )

        response = await self.provider.complete(
            CompletionRequest(
                messages=[
                    {"role": "system", "content":
                        "You are an expert prompt engineer specializing in "
                        "cross-model prompt adaptation. Your task is to adapt "
                        "a prompt from one LLM to another based on learned "
                        "transfer effects. Output ONLY the optimized prompt, "
                        "nothing else."},
                    {"role": "user", "content": adapter_prompt},
                ],
                model=self.model_id,
                temperature=0.0,
                max_tokens=8192,
            )
        )

        return response.content.strip()

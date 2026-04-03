from identa.calibration.types import CalibrationPair
from identa.transfer.types import TransferableKnowledge
from identa.providers.base import ModelProvider, CompletionRequest
from identa._internal.prompts.mapping_extractor import (
    MAPPING_EXTRACTOR_SYSTEM_PROMPT,
    build_mapping_extractor_user_prompt,
)

class MappingExtractor:
    """Extract transferable knowledge from calibrated prompt pairs."""

    def __init__(self, provider: ModelProvider, model_id: str):
        self.provider = provider
        self.model_id = model_id

    async def extract(
        self,
        pairs: list[CalibrationPair],
        source_model: str,
        target_model: str,
    ) -> TransferableKnowledge:
        user_prompt = build_mapping_extractor_user_prompt(
            pairs=pairs,
            source_model=source_model,
            target_model=target_model,
        )

        response = await self.provider.complete(
            CompletionRequest(
                messages=[
                    {"role": "system", "content": MAPPING_EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_id,
                temperature=0.0,
                max_tokens=8192,
            )
        )

        return TransferableKnowledge(
            source_model=source_model,
            target_model=target_model,
            summary=response.content,
            num_alignment_tasks=len(pairs),
        )

import asyncio
import logging
from identa.transfer.mapping_extractor import MappingExtractor
from identa.transfer.adapter import PromptAdapter
from identa.transfer.knowledge import KnowledgeStore
from identa.transfer.types import *
from identa.calibration.types import CalibrationPair
from identa.parser.base import PromptTemplate
from identa.providers.base import ModelProvider
from identa.config.settings import TransferConfig

logger = logging.getLogger(__name__)

class PromptBridgeEngine:
    """Cross-model prompt transfer via learned mapping."""

    def __init__(
        self,
        config: TransferConfig,
        extractor_provider: ModelProvider,
        extractor_model_id: str,
        adapter_provider: ModelProvider,
        adapter_model_id: str,
        knowledge_store: KnowledgeStore | None = None,
    ):
        self.config = config
        self.extractor = MappingExtractor(
            provider=extractor_provider,
            model_id=extractor_model_id,
        )
        self.adapter = PromptAdapter(
            provider=adapter_provider,
            model_id=adapter_model_id,
        )
        self.knowledge_store = knowledge_store

    async def learn_mapping(
        self,
        calibration_pairs: list[CalibrationPair],
        source_model: str,
        target_model: str,
    ) -> TransferableKnowledge:
        cache_key = self._mapping_cache_key(source_model, target_model, calibration_pairs)
        if self.knowledge_store:
            cached = self.knowledge_store.get(cache_key)
            if cached:
                logger.info(f"Using cached transfer mapping: {cache_key}")
                return cached

        knowledge = await self.extractor.extract(
            pairs=calibration_pairs,
            source_model=source_model,
            target_model=target_model,
        )

        if self.knowledge_store:
            self.knowledge_store.put(cache_key, knowledge)

        return knowledge

    async def transfer_prompt(
        self,
        source_prompt: PromptTemplate,
        knowledge: TransferableKnowledge,
        source_model: str,
        target_model: str,
        task_context: str | None = None,
    ) -> TransferResult:
        import time
        start = time.monotonic()

        transferred_text = await self.adapter.adapt(
            source_prompt_text=source_prompt.to_flat_text(),
            transferable_knowledge=knowledge.summary,
            source_model=source_model,
            target_model=target_model,
            task_context=task_context,
        )

        latency = (time.monotonic() - start) * 1000

        return TransferResult(
            original_prompt_id=source_prompt.id,
            source_prompt_text=source_prompt.to_flat_text(),
            transferred_prompt_text=transferred_text,
            source_model=source_model,
            target_model=target_model,
            knowledge_used=knowledge.cache_key(),
            adapter_model=self.adapter.model_id,
            latency_ms=latency,
            tokens_used=0,  # Updated by adapter
        )

    async def migrate_batch(
        self,
        prompts: list[PromptTemplate],
        calibration_pairs: list[CalibrationPair],
        source_model: str,
        target_model: str,
        progress_callback=None,
    ) -> MigrationReport:
        import time
        start = time.monotonic()

        knowledge = await self.learn_mapping(
            calibration_pairs, source_model, target_model
        )

        results = []
        errors = []
        for i, prompt in enumerate(prompts):
            try:
                result = await self.transfer_prompt(
                    prompt, knowledge, source_model, target_model
                )
                results.append(result)
            except Exception as e:
                errors.append({
                    "prompt_id": prompt.id,
                    "prompt_name": prompt.name,
                    "error": str(e),
                })
                logger.error(f"Failed to transfer {prompt.name}: {e}")

            if progress_callback:
                progress_callback(i + 1, len(prompts))

        total_tokens = sum(r.tokens_used for r in results)
        elapsed = time.monotonic() - start

        return MigrationReport(
            source_model=source_model,
            target_model=target_model,
            total_prompts=len(prompts),
            successful=len(results),
            failed=len(errors),
            total_duration_seconds=elapsed,
            total_tokens=total_tokens,
            estimated_cost_usd=self._estimate_cost(total_tokens),
            results=results,
            errors=errors,
            knowledge=knowledge,
        )

    def _mapping_cache_key(self, source, target, pairs) -> str:
        import hashlib
        content = f"{source}:{target}:" + ":".join(
            p.task_id for p in sorted(pairs, key=lambda p: p.task_id)
        )
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    def _estimate_cost(self, total_tokens: int) -> float:
        return (total_tokens / 1_000_000) * 9

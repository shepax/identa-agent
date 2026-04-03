import asyncio
import logging
import time
from identa.providers.base import (
    CompletionRequest, ModelIdentifier, ModelResponse,
    with_retries,
)

logger = logging.getLogger(__name__)

class LiteLLMAdapter:
    provider_name = "litellm"

    def __init__(self, config):
        self.config = config

    def supports_model(self, model_id: str) -> bool:
        return True

    def list_models(self) -> list[ModelIdentifier]:
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        """Send a completion request with retry on transient failures."""
        import litellm
        start = time.monotonic()

        async def _call():
            return await litellm.acompletion(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop
            )

        response = await with_retries(_call, max_retries=3)  # AUDIT-FIX: 2.1
        latency = (time.monotonic() - start) * 1000
        choice = response.choices[0]
        # Guard null usage (litellm can return None usage for some backends)
        usage = getattr(response, "usage", None)
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            finish_reason=choice.finish_reason,
            raw_response=None,  # AUDIT-FIX: 3.2
        )

    async def complete_batch(
        self,
        requests: list[CompletionRequest],
        max_concurrency: int = 5
    ) -> list[ModelResponse]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _limited(req):
            async with semaphore:
                return await self.complete(req)

        # AUDIT-FIX: 2.2 — return_exceptions=True prevents one failure killing the batch
        results = await asyncio.gather(
            *[_limited(r) for r in requests], return_exceptions=True
        )
        responses: list[ModelResponse | None] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                responses.append(None)
            else:
                responses.append(result)
        return responses

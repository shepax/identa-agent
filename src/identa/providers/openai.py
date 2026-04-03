import asyncio
import logging
import time
from openai import AsyncOpenAI
from identa.providers.base import (
    CompletionRequest, ModelIdentifier, ModelResponse,
    with_retries,
)
from identa.providers.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class OpenAIProvider:
    provider_name = "openai"

    def __init__(self, api_key: str, **kwargs):
        self._client = AsyncOpenAI(api_key=api_key)
        self._limiter = TokenBucketRateLimiter(
            requests_per_minute=500,
            tokens_per_minute=150_000
        )

    def supports_model(self, model_id: str) -> bool:
        return model_id.startswith(("gpt-", "o1", "o3", "o4"))

    def list_models(self) -> list[ModelIdentifier]:
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        """Send a completion request with retry on transient failures."""
        await self._limiter.acquire(estimated_tokens=len(str(request.messages)) // 4)
        start = time.monotonic()

        async def _call():
            return await self._client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )

        response = await with_retries(_call, max_retries=3)  # AUDIT-FIX: 2.1
        latency = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        # AUDIT-FIX: 2.8 — Guard null usage field
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            finish_reason=choice.finish_reason,
            raw_response=None,  # AUDIT-FIX: 3.2 — don't store potentially sensitive raw payload
        )

    async def complete_batch(
        self,
        requests: list[CompletionRequest],
        max_concurrency: int = 5,
    ) -> list[ModelResponse | None]:
        """Run a batch of requests, returning None for individual failures."""
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

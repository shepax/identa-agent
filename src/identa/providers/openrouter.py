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

class OpenRouterProvider:
    provider_name = "openrouter"

    def __init__(self, api_key: str, **kwargs):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        # OpenRouter usually has higher limits or handles them for us, 
        # but we'll include a sensible default.
        self._limiter = TokenBucketRateLimiter(
            requests_per_minute=200,
            tokens_per_minute=100_000
        )

    def supports_model(self, model_id: str) -> bool:
        # OpenRouter supports almost everything. 
        # For simplicity, we'll assume any model requested through this provider is supported.
        return True
        
    def list_models(self) -> list[ModelIdentifier]:
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        await self._limiter.acquire(estimated_tokens=len(str(request.messages)) // 4)

        start = time.monotonic()
        
        # OpenRouter prefers certain headers for ranking/attribution
        extra_headers = {
            "HTTP-Referer": "https://github.com/identa-ia", 
            "X-Title": "Identa CLI",
        }

        async def _call():
            return await self._client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
                extra_headers=extra_headers
            )

        response = await with_retries(_call, max_retries=3)  # AUDIT-FIX: 2.1
        latency = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

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

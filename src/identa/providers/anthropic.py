import asyncio
import logging
import time
from anthropic import AsyncAnthropic
from identa.providers.base import (
    CompletionRequest, ModelIdentifier, ModelResponse,
    with_retries,
)
from identa.providers.rate_limiter import TokenBucketRateLimiter
from identa import ProviderError

logger = logging.getLogger(__name__)


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, api_key: str, **kwargs):
        self._client = AsyncAnthropic(api_key=api_key)
        self._limiter = TokenBucketRateLimiter(
            requests_per_minute=50,
            tokens_per_minute=40_000
        )

    def supports_model(self, model_id: str) -> bool:
        return model_id.startswith("claude")

    def list_models(self) -> list[ModelIdentifier]:
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        """Send a completion request with retry on transient failures."""
        await self._limiter.acquire(estimated_tokens=len(str(request.messages)) // 4)
        start = time.monotonic()

        # AUDIT-FIX: 2.5 — Collect ALL system messages (not just first)
        system_parts = [m["content"] for m in request.messages if m["role"] == "system"]
        system_prompt = "\n".join(system_parts)

        user_messages = [m for m in request.messages if m["role"] != "system"]
        # AUDIT-FIX: 2.5 — Guard empty user messages to avoid Anthropic API rejection
        if not user_messages:
            raise ProviderError(
                "Anthropic requires at least one non-system message in the request"
            )

        async def _call():
            return await self._client.messages.create(
                model=request.model,
                messages=user_messages,
                system=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop_sequences=request.stop or [],
            )

        response = await with_retries(_call, max_retries=3)  # AUDIT-FIX: 2.1
        latency = (time.monotonic() - start) * 1000

        text_content = "".join(b.text for b in response.content if hasattr(b, "text"))

        return ModelResponse(
            content=text_content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            finish_reason=response.stop_reason or "stop",
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

import asyncio
import logging
import time
from google import genai
from google.genai import types
from identa.providers.base import (
    CompletionRequest, ModelIdentifier, ModelResponse,
    with_retries,
)
from identa.providers.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class GoogleProvider:
    provider_name = "google"

    def __init__(self, api_key: str, **kwargs):
        self._client = genai.Client(api_key=api_key)
        self._limiter = TokenBucketRateLimiter(
            requests_per_minute=15,
            tokens_per_minute=32_000
        )

    def supports_model(self, model_id: str) -> bool:
        return "gemini" in model_id.lower()

    def list_models(self) -> list[ModelIdentifier]:
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        """Send a completion request with retry on transient failures."""
        await self._limiter.acquire(estimated_tokens=len(str(request.messages)) // 4)
        start = time.monotonic()

        # AUDIT-FIX: 2.4 — Use system_instruction instead of silently mapping "system" to "user"
        system_parts = [m["content"] for m in request.messages if m["role"] == "system"]
        system_instruction = "\n".join(system_parts) if system_parts else None

        contents = []
        for msg in request.messages:
            if msg["role"] == "system":
                continue  # handled via system_instruction param
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,  # AUDIT-FIX: 2.4
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            stop_sequences=request.stop,
        )

        async def _call():
            return await self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=config,
            )

        response = await with_retries(_call, max_retries=3)  # AUDIT-FIX: 2.1
        latency = (time.monotonic() - start) * 1000

        # AUDIT-FIX: 2.3 — Guard response.text (may raise if safety-filtered)
        try:
            content = response.text or ""
        except (ValueError, AttributeError):
            content = ""
            logger.warning(
                "Google response had no text content "
                "(possibly blocked by safety filters or returned no candidates)"
            )

        # AUDIT-FIX: 2.3 — Safe attribute access on usage_metadata
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return ModelResponse(
            content=content,
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            finish_reason="stop",
            raw_response=None,  # AUDIT-FIX: 3.2
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

import asyncio
import logging
import time
import httpx
from identa.providers.base import (
    CompletionRequest, ModelIdentifier, ModelResponse,
    with_retries,
)
from identa.providers.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class OllamaProvider:
    provider_name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", **kwargs):
        self.base_url = base_url.rstrip("/")
        # AUDIT-FIX: 2.9 — Add rate limiter to prevent OOM from concurrent local requests
        self._limiter = TokenBucketRateLimiter(
            requests_per_minute=30,   # Conservative limit for local GPU/CPU
            tokens_per_minute=50_000
        )

    def supports_model(self, model_id: str) -> bool:
        return True

    def list_models(self) -> list[ModelIdentifier]:
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        """Send a completion request with retry on transient failures."""
        await self._limiter.acquire(estimated_tokens=len(str(request.messages)) // 4)
        start = time.monotonic()

        async def _call():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": request.model,
                        "messages": request.messages,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens,
                            "stop": request.stop or []
                        }
                    },
                    timeout=120.0
                )
                resp.raise_for_status()
                return resp.json()

        data = await with_retries(_call, max_retries=3)  # AUDIT-FIX: 2.1
        latency = (time.monotonic() - start) * 1000

        return ModelResponse(
            content=data["message"]["content"],
            model=data["model"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
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

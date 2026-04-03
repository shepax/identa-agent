import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol, Callable, Awaitable, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelIdentifier:
    """Canonical model reference."""
    provider: str          # "openai", "anthropic", "ollama", etc.
    model_id: str          # Provider-specific ID: "gpt-4o", "claude-sonnet-4-6"
    display_name: str      # Human-friendly: "GPT-4o", "Claude Sonnet 4.6"
    family: str            # "gpt", "claude", "llama", "qwen", "gemma"
    is_local: bool = False # True for Ollama/vLLM models


@dataclass(frozen=True)
class ModelResponse:
    """Standardized response from any provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str     # "stop", "length", "tool_calls"
    raw_response: dict | None = None  # Provider-specific raw payload (sanitized)


@dataclass(frozen=True)
class CompletionRequest:
    """Standardized request to any provider."""
    messages: list[dict]       # [{"role": ..., "content": ...}]
    model: str                 # Provider-specific model ID
    temperature: float = 0.0
    max_tokens: int = 4096
    stop: list[str] | None = None


class ModelProvider(Protocol):
    """Protocol every provider adapter must implement."""

    @property
    def provider_name(self) -> str: ...

    def list_models(self) -> list[ModelIdentifier]: ...

    async def complete(self, request: CompletionRequest) -> ModelResponse: ...

    async def complete_batch(
        self,
        requests: list[CompletionRequest],
        max_concurrency: int = 5
    ) -> list[ModelResponse | None]: ...

    def supports_model(self, model_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# AUDIT-FIX: 2.1 — Reusable async retry with exponential backoff
# ---------------------------------------------------------------------------

async def with_retries(
    coro_factory: Callable[[], Awaitable[Any]],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Retry an async call with exponential backoff.

    Args:
        coro_factory: Zero-arg callable that returns a new awaitable each call.
        max_retries: Maximum number of retry attempts (not counting first try).
        base_delay: Base delay in seconds; doubles on each retry.

    Raises:
        ProviderError: After all retries are exhausted.
    """
    from identa import ProviderError  # local import to avoid circular dep

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            if not _is_retryable(e) or attempt == max_retries:
                raise ProviderError(
                    f"Provider call failed after {attempt + 1} attempt(s): {e}"
                ) from e
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Transient error (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)


def _is_retryable(e: Exception) -> bool:
    """Return True if the exception looks like a transient, retryable failure."""
    error_str = str(e).lower()
    retryable_signals = [
        "rate_limit", "ratelimit", "429",
        "500", "502", "503", "504",
        "timeout", "timed out", "connection",
        "server error", "service unavailable",
    ]
    return any(signal in error_str for signal in retryable_signals)

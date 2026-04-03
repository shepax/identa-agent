import asyncio
import time

class TokenBucketRateLimiter:
    """Dual-bucket rate limiter: requests/min and tokens/min."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._request_tokens = requests_per_minute
        self._token_tokens = tokens_per_minute
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1) -> None:
        """Acquire rate limit tokens, sleeping outside the lock to avoid starvation.

        AUDIT-FIX: 1.3 — Previous implementation held the lock during asyncio.sleep(),
        serializing all coroutines through a 100ms polling loop. New implementation
        checks atomically under the lock, then sleeps OUTSIDE it so other
        coroutines can check/acquire concurrently.
        """
        while True:
            async with self._lock:
                self._refill()
                if self._request_tokens >= 1 and self._token_tokens >= estimated_tokens:
                    self._request_tokens -= 1
                    self._token_tokens -= estimated_tokens
                    return  # AUDIT-FIX: 1.3 — early return releases lock immediately
            # Sleep OUTSIDE the lock so other waiters can proceed
            await asyncio.sleep(0.05)  # AUDIT-FIX: 1.3 — reduced from 0.1s

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._request_tokens = min(
            self._rpm,
            self._request_tokens + elapsed * (self._rpm / 60)
        )
        self._token_tokens = min(
            self._tpm,
            self._token_tokens + elapsed * (self._tpm / 60)
        )
        self._last_refill = now

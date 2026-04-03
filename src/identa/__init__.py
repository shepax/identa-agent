class IdentaError(Exception):
    """Base exception for all Identa errors."""

class ConfigError(IdentaError):
    """Invalid or missing configuration."""

class ProviderError(IdentaError):
    """Model provider API errors (auth, rate limit, timeout)."""

class ProviderAuthError(ProviderError):
    """Missing or invalid API key."""

class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded — includes retry_after_seconds."""
    def __init__(self, message: str, retry_after_seconds: float = 60):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

class ParserError(IdentaError):
    """Failed to parse prompt file."""

class CalibrationError(IdentaError):
    """Calibration engine failure."""

class TransferError(IdentaError):
    """Transfer/migration failure."""

class TaskNotFoundError(IdentaError):
    """Requested alignment task not found."""

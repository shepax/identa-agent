import os
import sys
import logging
from pathlib import Path
from pydantic import SecretStr  # AUDIT-FIX: 3.1
from dotenv import load_dotenv
from identa.config.settings import IdentaSettings, ProviderConfig

logger = logging.getLogger(__name__)


def load_config(path: str | None = None) -> IdentaSettings:
    """Load configuration from environment and optional TOML file.

    Priority (highest wins): environment variables > TOML file > defaults.
    Walks up from CWD looking for identa.toml when no path is provided.
    """
    load_dotenv()
    settings = IdentaSettings()

    # AUDIT-FIX: 2.7 — Actually parse TOML instead of stub pass
    config_path = Path(path) if path else _find_config_file()
    if config_path and config_path.exists():
        try:
            toml_data = _load_toml(config_path)
            if "calibration" in toml_data:
                settings.calibration = settings.calibration.model_copy(
                    update=toml_data["calibration"]
                )
            if "scorer" in toml_data:
                settings.scorer = settings.scorer.model_copy(
                    update=toml_data["scorer"]
                )
            if "store" in toml_data:
                settings.store = settings.store.model_copy(
                    update=toml_data["store"]
                )
            if "transfer" in toml_data:
                settings.transfer = settings.transfer.model_copy(
                    update=toml_data["transfer"]
                )
            logger.debug(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to parse {config_path}: {e}. Using defaults.")

    # AUDIT-FIX: 3.1 — Wrap env vars in SecretStr so they never appear in logs/repr
    settings.providers.openai_api_key = SecretStr(
        os.getenv("OPENAI_API_KEY", settings.providers.openai_api_key.get_secret_value())
    )
    settings.providers.anthropic_api_key = SecretStr(
        os.getenv("ANTHROPIC_API_KEY", settings.providers.anthropic_api_key.get_secret_value())
    )
    settings.providers.google_api_key = SecretStr(
        os.getenv("GOOGLE_API_KEY", settings.providers.google_api_key.get_secret_value())
    )
    settings.providers.mistral_api_key = SecretStr(
        os.getenv("MISTRAL_API_KEY", settings.providers.mistral_api_key.get_secret_value())
    )
    settings.providers.openrouter_api_key = SecretStr(
        os.getenv("OPENROUTER_API_KEY", settings.providers.openrouter_api_key.get_secret_value())
    )
    settings.providers.ollama_base_url = os.getenv(
        "OLLAMA_BASE_URL", settings.providers.ollama_base_url
    )

    return settings


def _find_config_file() -> Path | None:
    """Walk up from CWD looking for identa.toml."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        candidate = parent / "identa.toml"
        if candidate.exists():
            return candidate
    return None


def _load_toml(path: Path) -> dict:
    """Load a TOML file using tomllib (3.11+) or tomli fallback."""
    if sys.version_info >= (3, 11):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        try:
            import tomli
            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            logger.warning(
                "tomli not installed and Python < 3.11. "
                "Install tomli for TOML config support: pip install tomli"
            )
            return {}

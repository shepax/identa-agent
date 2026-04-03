from pydantic import BaseModel, Field, SecretStr  # AUDIT-FIX: 3.1


class CalibrationConfig(BaseModel):
    """MAP-RPE hyperparameters — defaults from the paper."""
    target_model: str | None = None
    reflection_model: str | None = None
    calibration_questions: int = Field(20, ge=5, le=100)
    global_iterations: int = Field(20, ge=1, le=100)
    local_evolution_steps: int = Field(10, ge=1, le=50)
    prompt_archive_size: int = Field(1000, ge=50)
    num_islands: int = Field(3, ge=1, le=10)
    exploitation_ratio: float = Field(0.7, ge=0.0, le=1.0)
    exploration_ratio: float = Field(0.2, ge=0.0, le=1.0)
    elite_selection_ratio: float = Field(0.1, ge=0.0, le=1.0)
    migration_interval: int = Field(50, ge=10)
    migration_rate: float = Field(0.1, ge=0.0, le=1.0)
    performance_weight: float = Field(0.8, ge=0.0, le=1.0)  # lambda in paper


class TransferConfig(BaseModel):
    """PromptBridge transfer settings."""
    mapping_extractor_model: str = "gpt-5"       # Best per ablation study
    adapter_model: str = "gpt-5"                  # Best per ablation study
    fallback_extractor_model: str = "claude-sonnet-4-6"
    fallback_adapter_model: str = "claude-sonnet-4-6"
    max_alignment_tasks: int = Field(54, ge=5)    # Paper uses 54


class ScorerConfig(BaseModel):
    """Behavioral scoring weights — configurable per domain."""
    syntax_validity_weight: float = 0.35
    entry_point_weight: float = 0.35
    risk_free_weight: float = 0.20
    no_undesirable_weight: float = 0.10
    # Domain overrides
    domain_overrides: dict[str, dict[str, float]] = {}


class ProviderConfig(BaseModel):
    """Provider-specific settings."""
    # AUDIT-FIX: 3.1 — Use SecretStr to prevent accidental key exposure in logs/repr
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    mistral_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    ollama_base_url: str = "http://localhost:11434"
    default_temperature: float = 0.0
    max_retries: int = 3
    timeout_seconds: int = 120


class StoreConfig(BaseModel):
    """Prompt store settings."""
    backend: str = "sqlite"  # "sqlite" or "postgres"
    sqlite_path: str = "~/.identa/store.db"
    postgres_url: str = ""


class IdentaSettings(BaseModel):
    """Top-level configuration — maps to identa.toml."""
    calibration: CalibrationConfig = CalibrationConfig()
    transfer: TransferConfig = TransferConfig()
    scorer: ScorerConfig = ScorerConfig()
    providers: ProviderConfig = ProviderConfig()
    store: StoreConfig = StoreConfig()
    log_level: str = "INFO"
    cache_dir: str = "~/.identa/cache"
    telemetry_enabled: bool = False

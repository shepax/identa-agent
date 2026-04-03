from identa.providers.base import ModelProvider, ModelIdentifier

# Known model families and their canonical IDs
MODEL_CATALOG: dict[str, ModelIdentifier] = {
    # OpenAI
    "gpt-4o": ModelIdentifier("openai", "gpt-4o", "GPT-4o", "gpt"),
    "gpt-5": ModelIdentifier("openai", "gpt-5", "GPT-5", "gpt"),
    "o3": ModelIdentifier("openai", "o3", "o3", "gpt"),
    "o4-mini": ModelIdentifier("openai", "o4-mini", "o4-mini", "gpt"),
    # Anthropic
    "claude-opus-4-6": ModelIdentifier("anthropic", "claude-opus-4-6", "Claude Opus 4.6", "claude"),
    "claude-sonnet-4-6": ModelIdentifier("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6", "claude"),
    # Meta
    "llama-3.1-70b": ModelIdentifier("ollama", "llama3.1:70b", "Llama 3.1 70B", "llama"),
    "llama-3.1-405b": ModelIdentifier("ollama", "llama3.1:405b", "Llama 3.1 405B", "llama"),
    # Qwen
    "qwen3-32b": ModelIdentifier("ollama", "qwen3:32b", "Qwen3 32B", "qwen"),
    # Google
    "gemini-2.5-pro": ModelIdentifier("google", "gemini-2.5-pro", "Gemini 2.5 Pro", "gemini"),
    # Mistral
    "mistral-large": ModelIdentifier("mistral", "mistral-large-latest", "Mistral Large", "mistral"),
}

class ProviderRegistry:
    """Resolves model IDs to providers and manages provider lifecycle."""

    def __init__(self, config):
        self._providers: dict[str, ModelProvider] = {}
        self._config = config
        self._init_providers()

    def _init_providers(self):
        from identa.providers.openai import OpenAIProvider
        from identa.providers.anthropic import AnthropicProvider
        from identa.providers.google import GoogleProvider
        from identa.providers.ollama import OllamaProvider
        from identa.providers.openrouter import OpenRouterProvider
        
        cfg = self._config.providers
        
        # AUDIT-FIX: 3.1 — .get_secret_value() exposes the raw string only at the call site
        openai_key = cfg.openai_api_key.get_secret_value()
        if openai_key:
            self._providers["openai"] = OpenAIProvider(api_key=openai_key)
            
        anthropic_key = cfg.anthropic_api_key.get_secret_value()
        if anthropic_key:
            self._providers["anthropic"] = AnthropicProvider(api_key=anthropic_key)
            
        google_key = cfg.google_api_key.get_secret_value()
        if google_key:
            self._providers["google"] = GoogleProvider(api_key=google_key)

        openrouter_key = cfg.openrouter_api_key.get_secret_value()
        if openrouter_key:
            self._providers["openrouter"] = OpenRouterProvider(api_key=openrouter_key)
            
        # Ollama is always available locally by default
        self._providers["ollama"] = OllamaProvider(base_url=cfg.ollama_base_url)


    def resolve(self, model_id: str) -> tuple[ModelProvider, ModelIdentifier]:
        """Given a user-facing model ID, return the provider + model info."""
        # 1. Check for explicit provider prefix: "openai/gpt-4o"
        if "/" in model_id:
            parts = model_id.split("/")
            first_part = parts[0]
            if first_part in self._providers:
                actual_model = "/".join(parts[1:])
                return self._providers[first_part], ModelIdentifier(first_part, actual_model, actual_model, parts[-1])

        # 2. Check canonical catalog entries
        if model_id in MODEL_CATALOG:
            ident = MODEL_CATALOG[model_id]
            provider = self._providers.get(ident.provider)
            if provider:
                return provider, ident
            
        # 3. Fallback to OpenRouter if configured (user preference)
        if "openrouter" in self._providers:
            provider = self._providers["openrouter"]
            ident = ModelIdentifier("openrouter", model_id, model_id, "unknown")
            return provider, ident

        # 4. Final fallback: try LiteLLM universal adapter
        from identa.providers.litellm_adapter import LiteLLMAdapter
        adapter = LiteLLMAdapter(self._config)
        ident = ModelIdentifier("litellm", model_id, model_id, "unknown")
        return adapter, ident

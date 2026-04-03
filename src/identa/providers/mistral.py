from identa.providers.base import *

class MistralProvider:
    provider_name = "mistral"
    def list_models(self) -> list[ModelIdentifier]: return []
    def supports_model(self, model_id: str) -> bool: return "mistral" in model_id
    async def complete(self, request: CompletionRequest) -> ModelResponse: pass
    async def complete_batch(self, requests: list[CompletionRequest], max_concurrency: int = 5) -> list[ModelResponse]: pass

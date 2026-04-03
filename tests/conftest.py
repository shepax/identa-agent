import pytest
import asyncio
from identa.providers.base import ModelProvider, ModelResponse, CompletionRequest

class MockProvider:
    """Deterministic mock provider for unit/integration tests."""
    provider_name = "mock"

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._default_response = "Mock response"
        self.call_log: list[CompletionRequest] = []

    def supports_model(self, model_id: str) -> bool:
        return True

    def list_models(self):
        return []

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        self.call_log.append(request)
        
        last_msg = request.messages[-1]["content"].lower()
        if "def " in last_msg or "code" in last_msg:
            content = "def add(a, b):\n    return a + b"
        elif "improve" in last_msg or "reflect" in last_msg:
            content = "You are a helpful coding assistant. Answer the user's question concisely."
        elif "summarize" in last_msg or "difference" in last_msg:
            content = "Target model prefers direct answers. Source model was more chatty."
        elif "adapt" in last_msg or "transfer" in last_msg:
            content = "Adapted Target Prompt: concise code without pleasantries."
        else:
            content = self._responses.get(
                request.messages[-1]["content"][:50],
                self._default_response
            )
            
        await asyncio.sleep(0.01) # Simulate tiny latency
        
        return ModelResponse(
            content=content,
            model=request.model,
            input_tokens=len(str(request.messages)) // 4,
            output_tokens=len(content) // 4,
            latency_ms=10.0,
            finish_reason="stop",
        )

    async def complete_batch(self, requests, max_concurrency=5):
        return [await self.complete(r) for r in requests]

@pytest.fixture
def mock_provider():
    return MockProvider()

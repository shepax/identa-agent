import asyncio
import sys
import logging
from unittest.mock import MagicMock
from identa.config.settings import CalibrationConfig, ScorerConfig
from identa.calibration.engine import MAPRPEEngine
from identa.calibration.scorer import BehavioralScorer
from identa.tasks.schema import AlignmentTask, TaskInstance
from identa.providers.base import ModelProvider, ModelResponse, CompletionRequest

# --- Mock Infrastructure ---

class SophisticatedMockProvider(ModelProvider):
    provider_name = "sophisticated_mock"

    def __init__(self):
        self._iteration = 0

    def supports_model(self, model_id: str) -> bool:
        return True

    async def complete(self, request: CompletionRequest) -> ModelResponse:
        self._iteration += 1
        prompt = request.messages[-1]["content"].lower()
        
        # Simulate improvement over time for MAP-RPE
        if "expert coding assistant" in prompt:
            # Better performance if the refined prompt is used
            perf = 0.95
            content = "def reverse_integer(x):\n    # Optimized implementation\n    return int(str(abs(x))[::-1]) * (1 if x >= 0 else -1)"
        elif "generation 0" in prompt or "iteration 0" in prompt:
            perf = 0.5
            content = "Here is the code: def rev(x): return x"
        else:
            perf = 0.7 + (min(self._iteration, 100) / 400.0) # Slow drift upwards
            content = f"Simulated response (iteration {self._iteration})"

        return ModelResponse(
            content=content,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            latency_ms=5.0,
            finish_reason="stop",
            raw_response={"perf_sim": perf} # Internal hint
        )

    async def complete_batch(self, requests, max_concurrency=5):
        return [await self.complete(r) for r in requests]

# --- Verification Logic ---

async def verify_e2e():
    print("🚀 Starting Zero-Cost E2E Verification...")
    
    # 1. Setup Config
    config = CalibrationConfig(
        global_iterations=3,
        local_evolution_steps=2,
        num_islands=2,
        calibration_questions=5
    )
    scorer_cfg = ScorerConfig()
    scorer = BehavioralScorer(scorer_cfg)
    
    # 2. Setup Mock Providers
    provider = SophisticatedMockProvider()
    
    # 3. Setup Engine
    # We'll monkeypatch the performance evaluator for deterministic results
    engine = MAPRPEEngine(
        config=config,
        target_provider=provider,
        target_model_id="mock-gpt-4",
        reflection_provider=provider,
        reflection_model_id="mock-gpt-4",
        scorer=scorer
    )
    
    # Mock performance evaluation to be deterministic based on provider hint
    original_eval = engine._evaluate_performance
    def mock_eval(response, instance, metric):
        # If the provider returned a real-ish response, give high score
        if "def " in response:
            return 0.95
        return 0.4
    engine._evaluate_performance = mock_eval

    # 4. Setup Task
    task = AlignmentTask(
        task_id="e2e_mock_task",
        name="Mock Coding Task",
        domain="coding",
        description="Verify engine loops without API cost",
        instances=[
            TaskInstance(question="Reverse an integer", answer="0"),
            TaskInstance(question="Sum two numbers", answer="3"),
            TaskInstance(question="Find prime factors", answer="[2, 3]"),
            TaskInstance(question="Sort a list", answer="[1, 2, 3]"),
            TaskInstance(question="Check palindrome", answer="True"),
        ],
        evaluation_metric="functional_correctness",
        source="builtin"
    )

    # 5. Run Calibration
    def on_progress(g, t, score):
        print(f"  [Progress] Generation {g}/{t} | Best Score: {score:.3f}")

    print("🛠 Running MAP-RPE Engine...")
    result = await engine.calibrate(task, "You are a assistant.", on_progress)
    
    print("\n✅ Verification Complete!")
    print(f"🎯 Optimal Prompt: {result.optimal_prompt[:50]}...")
    print(f"📊 Final Score: {result.combined_score:.3f}")
    print(f"📞 API Calls Simulated: {result.total_api_calls}")
    print(f"⏱ Duration: {result.duration_seconds:.2f}s")
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    success = asyncio.run(verify_e2e())
    sys.exit(0 if success else 1)

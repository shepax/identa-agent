import asyncio
from dataclasses import dataclass
from identa.providers.base import ModelProvider, CompletionRequest
from identa.tasks.schema import AlignmentTask, TaskInstance

@dataclass
class DriftReport:
    source_model: str
    target_model: str
    prompt_id: str
    transfer_gap: float           # Δ(Ms→Mt, T) — negative means degradation
    source_performance: float     # A(Ms, T, p*Ms)
    target_direct_performance: float  # A(Mt, T, p*Ms) — source prompt on target
    target_optimal_estimate: float    # Estimated A(Mt, T, p*Mt)
    num_samples: int
    per_sample_results: list[dict]
    risk_level: str               # "low", "medium", "high", "critical"
    std_dev: float = 0.0          # Variance in performance

class DriftAnalyzer:
    """Quantify model drift for a given prompt.

    Measures the transfer gap: how much performance drops when
    a source-optimized prompt is used directly on the target model.
    """

    def __init__(
        self,
        source_provider: ModelProvider,
        source_model_id: str,
        target_provider: ModelProvider,
        target_model_id: str,
    ):
        self.source_provider = source_provider
        self.source_model_id = source_model_id
        self.target_provider = target_provider
        self.target_model_id = target_model_id

    async def analyze(
        self,
        prompt_text: str,
        task: AlignmentTask,
        num_samples: int = 20,
    ) -> DriftReport:
        import random
        samples = random.sample(
            task.instances,
            min(num_samples, len(task.instances))
        )

        source_results = await self._evaluate_model(
            self.source_provider, self.source_model_id,
            prompt_text, samples, task.evaluation_metric
        )
        target_results = await self._evaluate_model(
            self.target_provider, self.target_model_id,
            prompt_text, samples, task.evaluation_metric
        )

        source_avg = sum(r["score"] for r in source_results) / len(source_results)
        target_avg = sum(r["score"] for r in target_results) / len(target_results)
        gap = target_avg - source_avg  

        # Simple standard deviation calculation
        import math
        variances = [(r["score"] - target_avg)**2 for r in target_results]
        std_dev = math.sqrt(sum(variances) / len(variances)) if variances else 0.0

        if gap > -0.05:
            risk = "low"
        elif gap > -0.15:
            risk = "medium"
        elif gap > -0.30:
            risk = "high"
        else:
            risk = "critical"

        return DriftReport(
            source_model=self.source_model_id,
            target_model=self.target_model_id,
            prompt_id="analyzed_prompt",
            transfer_gap=gap,
            source_performance=source_avg,
            target_direct_performance=target_avg,
            target_optimal_estimate=0.0,
            num_samples=len(samples),
            per_sample_results=[
                {"source": s, "target": t}
                for s, t in zip(source_results, target_results)
            ],
            risk_level=risk,
            std_dev=std_dev,
        )

    async def _evaluate_model(
        self, provider, model_id, prompt_text, samples, metric
    ) -> list[dict]:
        results = []
        for instance in samples:
            full_prompt = f"{prompt_text}\n\n{instance.question}"
            response = await provider.complete(
                CompletionRequest(
                    messages=[{"role": "user", "content": full_prompt}],
                    model=model_id,
                    temperature=0.0,
                )
            )
            score = self._score(response.content, instance.answer, metric)
            results.append({
                "question": instance.question[:100],
                "score": score,
                "response_length": len(response.content),
            })
        return results

    def _score(self, response, answer, metric) -> float:
        if metric == "exact_match":
            return 1.0 if response.strip() == answer.strip() else 0.0
        from difflib import SequenceMatcher
        return SequenceMatcher(None, response.lower(), answer.lower()).ratio()

import asyncio
import logging
from dataclasses import dataclass
from identa.providers.base import ModelProvider, CompletionRequest
from identa.tasks.schema import AlignmentTask, TaskInstance
from identa.calibration.scorer import BehavioralScorer
from identa.config.settings import IdentaSettings

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    source_model: str
    target_model: str
    source_perf: float
    source_behavior: float
    target_perf: float
    target_behavior: float
    performance_gap: float
    behavior_gap: float
    total_gap: float

class DriftEstimator:
    """Measures the performance difference (drift) of a prompt between two models."""

    def __init__(self, config: IdentaSettings, source_provider: ModelProvider, 
                 source_model_id: str, target_provider: ModelProvider, 
                 target_model_id: str, scorer: BehavioralScorer):
        self.config = config
        self.source_provider = source_provider
        self.source_model_id = source_model_id
        self.target_provider = target_provider
        self.target_model_id = target_model_id
        self.scorer = scorer

    async def estimate(self, prompt: str, task: AlignmentTask, samples: int = 10) -> DriftResult:
        import random
        instances = random.sample(task.instances, min(samples, len(task.instances)))

        source_perf_scores = []
        source_behavior_scores = []
        target_perf_scores = []
        target_behavior_scores = []

        async def eval_model(provider, model_id, inst):
            full_prompt = self._apply_template(prompt, inst.question)
            try:
                response = await provider.complete(CompletionRequest(
                    messages=[{"role": "user", "content": full_prompt}],
                    model=model_id,
                    temperature=0.0
                ))
                perf = self._evaluate_performance(response.content, inst, task.evaluation_metric)
                behav = self.scorer.score(response.content, inst).total
                return perf, behav
            except Exception as e:
                logger.error(f"Error evaluating model {model_id}: {e}")
                return 0.0, 0.0

        # Evaluate on all sample instances
        for inst in instances:
            s_perf, s_behav = await eval_model(self.source_provider, self.source_model_id, inst)
            t_perf, t_behav = await eval_model(self.target_provider, self.target_model_id, inst)

            source_perf_scores.append(s_perf)
            source_behavior_scores.append(s_behav)
            target_perf_scores.append(t_perf)
            target_behavior_scores.append(t_behav)

        s_perf_avg = sum(source_perf_scores) / len(source_perf_scores)
        s_behav_avg = sum(source_behavior_scores) / len(source_behavior_scores)
        t_perf_avg = sum(target_perf_scores) / len(target_perf_scores)
        t_behav_avg = sum(target_behavior_scores) / len(target_behavior_scores)

        perf_gap = s_perf_avg - t_perf_avg
        behav_gap = s_behav_avg - t_behav_avg
        
        # Combined gap based on scorer weights
        w_perf = self.config.calibration.performance_weight
        total_gap = (w_perf * perf_gap) + ((1 - w_perf) * behav_gap)

        return DriftResult(
            source_model=self.source_model_id,
            target_model=self.target_model_id,
            source_perf=s_perf_avg,
            source_behavior=s_behav_avg,
            target_perf=t_perf_avg,
            target_behavior=t_behav_avg,
            performance_gap=perf_gap,
            behavior_gap=behav_gap,
            total_gap=total_gap
        )

    def _apply_template(self, prompt_template: str, question: str) -> str:
        if "{question}" in prompt_template:
            return prompt_template.replace("{question}", question)
        if "{task}" in prompt_template:
            return prompt_template.replace("{task}", question)
        return f"{prompt_template}\n\n{question}"

    def _evaluate_performance(self, response: str, instance: TaskInstance, metric: str) -> float:
        if metric == "exact_match":
            return 1.0 if response.strip() == instance.answer.strip() else 0.0
        elif metric == "text_similarity":
            from difflib import SequenceMatcher
            return SequenceMatcher(None, response.lower(), instance.answer.lower()).ratio()
        return 0.0

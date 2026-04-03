import asyncio
import logging
from identa.calibration.types import *
from identa.calibration.island import IslandManager
from identa.calibration.evolver import ReflectiveEvolver
from identa.calibration.scorer import BehavioralScorer
from identa.calibration.cache import CalibrationCache
from identa.config.settings import CalibrationConfig
from identa.providers.base import ModelProvider, CompletionRequest
from identa.tasks.schema import AlignmentTask, TaskInstance
from identa.tasks.domains import BUILTIN_DOMAINS
from identa.tasks.generator import QuestionGenerator

logger = logging.getLogger(__name__)

class MAPRPEEngine:
    """Model-Adaptive Reflective Prompt Evolution.

    Implements Algorithm 1 from the PromptBridge paper.
    Produces task- and model-specific optimal prompts through
    iterative reflective refinement with island-based population evolution.
    """

    def __init__(
        self,
        config: CalibrationConfig,
        target_provider: ModelProvider,
        target_model_id: str,
        reflection_provider: ModelProvider,
        reflection_model_id: str,
        scorer: BehavioralScorer,
        cache: CalibrationCache | None = None,
    ):
        self.config = config
        self.target_provider = target_provider
        self.target_model_id = target_model_id
        self.reflection_provider = reflection_provider
        self.reflection_model_id = reflection_model_id
        self.scorer = scorer
        self.cache = cache

        self.island_manager = IslandManager(
            num_islands=config.num_islands,
            archive_size=config.prompt_archive_size,
            migration_interval=config.migration_interval,
            migration_rate=config.migration_rate,
        )
        self.evolver = ReflectiveEvolver(
            reflection_provider=reflection_provider,
            reflection_model_id=reflection_model_id,
            exploitation_ratio=config.exploitation_ratio,
            exploration_ratio=config.exploration_ratio,
            elite_selection_ratio=config.elite_selection_ratio,
        )

    async def calibrate(
        self,
        task: AlignmentTask,
        source_prompt: str | None = None,
        progress_callback=None,
    ) -> CalibrationResult:
        # AUDIT-FIX: 2.6 — Guard empty task instances before entering the O(n²) loop
        if not task.instances:
            raise CalibrationError(
                f"Task '{task.task_id}' has no instances. "
                "Provide at least one question-answer pair."
            )

        cache_key = f"{self.target_model_id}:{task.task_id}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached calibration for {cache_key}")
                return cached

        initial_prompt = source_prompt or task.default_prompt
        self.island_manager.initialize(initial_prompt)
        global_best = PromptCandidate(prompt_text=initial_prompt)

        import random
        instances = random.sample(
            task.instances,
            min(self.config.calibration_questions, len(task.instances))
        )

        total_api_calls = 0
        total_tokens = 0
        start_time = asyncio.get_event_loop().time()

        for g in range(self.config.global_iterations):
            for j, instance in enumerate(instances):
                try:  # AUDIT-FIX: 2.1 / 7.2 — graceful degradation on provider error
                    current_best = self.island_manager.global_best
                    prompt_text = self._apply_template(
                        current_best.prompt_text, instance.question
                    )

                    logger.debug(f"[Gen {g} | Island {self.island_manager.current_island_idx}] Sending Prompt: {prompt_text!r}")
                    response = await self.target_provider.complete(
                        CompletionRequest(
                            messages=[{"role": "user", "content": prompt_text}],
                            model=self.target_model_id,
                            temperature=0.0,
                        )
                    )
                    total_api_calls += 1
                    total_tokens += response.input_tokens + response.output_tokens
                    logger.debug(f"Raw Response: {response.content[:150]}...")

                    perf_score = self._evaluate_performance(
                        response.content, instance, task.evaluation_metric
                    )
                    behav_score = self.scorer.score(response.content, instance)

                    if perf_score >= 1.0:
                        continue

                    candidate = PromptCandidate(
                        prompt_text=current_best.prompt_text,
                        performance_score=perf_score,
                        behavioral_score=behav_score.total,
                        combined_score=(
                            self.config.performance_weight * perf_score +
                            (1 - self.config.performance_weight) * behav_score.total
                        ),
                        generation=g,
                    )
                    self.island_manager.add_to_current_island(candidate)

                    for l in range(self.config.local_evolution_steps):
                        parent = self.island_manager.select_parent()
                        child_prompt = await self.evolver.evolve(
                            parent=parent,
                            instance=instance,
                            task=task,
                            evaluation_feedback={
                                "performance": perf_score,
                                "behavioral": behav_score,
                                "response": response.content[:500],
                                "expected": instance.answer[:500],
                            }
                        )
                        total_api_calls += 1
                        logger.debug(f"Reflected Child Prompt: {child_prompt!r}")

                        child_response = await self.target_provider.complete(
                            CompletionRequest(
                                messages=[{"role": "user", "content":
                                    self._apply_template(child_prompt, instance.question)}],
                                model=self.target_model_id,
                                temperature=0.0,
                            )
                        )
                        total_api_calls += 1
                        total_tokens += child_response.input_tokens + child_response.output_tokens
                        logger.debug(f"Child Raw Response: {child_response.content[:150]}...")

                        child_perf = self._evaluate_performance(
                            child_response.content, instance, task.evaluation_metric
                        )
                        child_behav = self.scorer.score(child_response.content, instance)
                        child_combined = (
                            self.config.performance_weight * child_perf +
                            (1 - self.config.performance_weight) * child_behav.total
                        )

                        child_candidate = PromptCandidate(
                            prompt_text=child_prompt,
                            performance_score=child_perf,
                            behavioral_score=child_behav.total,
                            combined_score=child_combined,
                            generation=g,
                            parent_id=parent.id,
                        )
                        self.island_manager.add_to_current_island(child_candidate)
                        self.island_manager.maybe_migrate()

                except Exception as e:  # AUDIT-FIX: 2.1 / 7.2 — skip bad instances, log and continue
                    from identa import ProviderError
                    if isinstance(e, ProviderError):
                        logger.warning(f"Skipping instance in gen {g} due to provider error: {e}")
                    else:
                        logger.warning(f"Unexpected error in gen {g}, instance {j}: {e}")
                    continue

            self.island_manager.update_global_best()

            if progress_callback:
                progress_callback(g + 1, self.config.global_iterations,
                                  self.island_manager.global_best.combined_score)

        elapsed = asyncio.get_event_loop().time() - start_time
        best = self.island_manager.global_best
        result = CalibrationResult(
            model_id=self.target_model_id,
            task_id=task.task_id,
            optimal_prompt=best.prompt_text,
            performance_score=best.performance_score,
            behavioral_score=best.behavioral_score,
            combined_score=best.combined_score,
            iterations_used=self.config.global_iterations,
            total_api_calls=total_api_calls,
            total_tokens=total_tokens,
            duration_seconds=elapsed,
        )

        if self.cache:
            self.cache.put(cache_key, result)

        return result

    async def calibrate_from_domain(
        self,
        domain_name: str,
        initial_prompt: str,
        use_agent_generation: bool = False,
        num_questions: int = 10,
        progress_callback=None,
    ) -> CalibrationResult:
        """New entry point for domain-based calibration."""
        if domain_name not in BUILTIN_DOMAINS:
            raise ValueError(f"Domain '{domain_name}' not found.")
            
        domain = BUILTIN_DOMAINS[domain_name]
        
        if use_agent_generation:
            generator = QuestionGenerator(self.reflection_provider, self.reflection_model_id)
            instances = await generator.generate_questions(domain, count=num_questions)
        else:
            import random
            instances = random.sample(
                domain.static_questions, 
                min(num_questions, len(domain.static_questions))
            )

        task = AlignmentTask(
            task_id=f"domain_{domain_name}",
            name=domain.name,
            domain=domain_name,
            description=domain.description,
            instances=instances,
            evaluation_metric="text_similarity",
            source="builtin",
            default_prompt=initial_prompt,
            generation_mode="agent" if use_agent_generation else "static"
        )
        
        return await self.calibrate(task, initial_prompt, progress_callback)

    def _apply_template(self, prompt_template: str, question: str) -> str:
        if "{question}" in prompt_template:
            return prompt_template.replace("{question}", question)
        if "{task}" in prompt_template:
            return prompt_template.replace("{task}", question)
        return f"{prompt_template}\n\n{question}"

    def _evaluate_performance(
        self, response: str, instance: 'TaskInstance', metric: str
    ) -> float:
        if metric == "exact_match":
            return 1.0 if response.strip() == instance.answer.strip() else 0.0
        elif metric == "text_similarity":
            return self._text_similarity(response, instance.answer)
        elif metric == "functional_correctness":
            return self._code_correctness(response, instance)
        else:
            return self._text_similarity(response, instance.answer)

    def _text_similarity(self, a: str, b: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _code_correctness(self, code: str, instance: 'TaskInstance') -> float:
        return self._text_similarity(code, instance.answer)

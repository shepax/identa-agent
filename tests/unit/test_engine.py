import pytest
import asyncio
from identa.calibration.engine import MAPRPEEngine
from identa.calibration.scorer import BehavioralScorer
from identa.config.settings import CalibrationConfig, ScorerConfig
from identa.tasks.schema import AlignmentTask, TaskInstance

@pytest.mark.asyncio
async def test_calibration_e2e_mock(mock_provider):
    """Verify that the MAP-RPE engine can run a full calibration loop with a mock provider."""
    config = CalibrationConfig(
        global_iterations=2,
        local_evolution_steps=2,
        calibration_questions=5,
        num_islands=2
    )
    
    scorer = BehavioralScorer(ScorerConfig(), domain="coding")
    
    engine = MAPRPEEngine(
        config=config,
        target_provider=mock_provider,
        target_model_id="mock-gpt",
        reflection_provider=mock_provider,
        reflection_model_id="mock-gpt",
        scorer=scorer
    )
    
    task = AlignmentTask(
        task_id="test_task",
        name="Test Task",
        domain="coding",
        description="Testing the engine",
        instances=[
            TaskInstance(question="Add 2 and 2", answer="4", metadata={"entry_point": "add"}),
            TaskInstance(question="Multiply 3 and 3", answer="9", metadata={"entry_point": "mul"})
        ],
        evaluation_metric="text_similarity",
        source="unit_test",
        default_prompt="You are a calculator."
    )
    
    result = await engine.calibrate(task)
    
    assert result.total_api_calls > 0
    assert result.optimal_prompt is not None
    assert result.combined_score >= 0.0
    assert len(mock_provider.call_log) > 5

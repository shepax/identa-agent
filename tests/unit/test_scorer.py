import pytest
from identa.calibration.scorer import BehavioralScorer
from identa.config.settings import ScorerConfig
from identa.tasks.schema import TaskInstance

@pytest.fixture
def scorer():
    return BehavioralScorer(ScorerConfig(), domain="coding")

@pytest.fixture
def task_instance():
    return TaskInstance(
        question="Write a function that adds two numbers",
        answer="def add(a, b): return a + b",
        metadata={"entry_point": "add"},
    )

class TestBehavioralScorer:
    def test_perfect_code_scores_high(self, scorer, task_instance):
        output = "def add(a, b):\n    return a + b"
        result = scorer.score(output, task_instance)
        assert result.syntax_validity == 1.0
        assert result.entry_point_defined == 1.0
        assert result.risk_free_patterns == 1.0
        assert result.total >= 0.9

    def test_syntax_error_penalized(self, scorer, task_instance):
        output = "def add(a, b)\n    return a + b"  # Missing colon
        result = scorer.score(output, task_instance)
        assert result.syntax_validity == 0.0
        assert result.total < 0.7

    def test_missing_entry_point_penalized(self, scorer, task_instance):
        output = "def sum_numbers(a, b):\n    return a + b"
        result = scorer.score(output, task_instance)
        assert result.entry_point_defined == 0.0

    def test_risky_patterns_penalized(self, scorer, task_instance):
        output = "def add(a, b):\n    exec('return a + b')"
        result = scorer.score(output, task_instance)
        assert result.risk_free_patterns < 1.0

    def test_code_in_markdown_block_extracted(self, scorer, task_instance):
        output = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
        result = scorer.score(output, task_instance)
        assert result.syntax_validity == 1.0
        assert result.entry_point_defined == 1.0

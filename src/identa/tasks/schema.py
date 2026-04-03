from dataclasses import dataclass, field

@dataclass(frozen=True)
class TaskInstance:
    """Single question-answer pair within an alignment task."""
    question: str
    answer: str
    metadata: dict = field(default_factory=dict)

@dataclass
class AlignmentTask:
    """A calibration task used to learn model-specific prompt preferences."""
    task_id: str                    # Unique identifier
    name: str                       # Human-readable name
    domain: str                     # "coding", "planning", "reasoning", "writing"
    description: str                # What this task evaluates
    instances: list[TaskInstance]    # Question-answer pairs
    evaluation_metric: str          # "exact_match", "functional_correctness",
                                    # "text_similarity", "custom"
    source: str                     # "builtin", "community", "custom"
    default_prompt: str = ""        # Initial prompt template for this task
    generation_mode: str = "static" # "static" | "agent"
    generation_prompt: str = ""     # Prompt for agent-based generation

    @property
    def num_instances(self) -> int:
        return len(self.instances)

    def is_agent_generated(self) -> bool:
        return self.generation_mode == "agent"

    def is_static(self) -> bool:
        return self.generation_mode == "static"

@dataclass
class DomainCalibrationSet:
    """A collection of questions and generation logic for a specific persona domain."""
    domain_id: str
    name: str
    description: str
    static_questions: list[TaskInstance]
    agent_generation_prompt: str
    tags: list[str] = field(default_factory=list)

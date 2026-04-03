from dataclasses import dataclass, field

@dataclass
class PromptCandidate:
    """A prompt variant in the evolutionary population."""
    prompt_text: str
    performance_score: float = 0.0     # Task accuracy (a_g in paper)
    behavioral_score: float = 0.0      # Structural quality (b_g in paper)
    combined_score: float = 0.0        # lambda * perf + (1-lambda) * behavior
    island_id: int = 0
    generation: int = 0
    parent_id: str | None = None
    evaluation_details: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Deterministic hash-based ID for deduplication."""
        import hashlib
        return hashlib.sha256(self.prompt_text.encode()).hexdigest()[:16]

@dataclass
class IslandState:
    """State of a single evolutionary island."""
    island_id: int
    population: list[PromptCandidate] = field(default_factory=list)
    best_candidate: PromptCandidate | None = None
    generation_counter: int = 0
    archive_size: int = 1000

    def add_candidate(self, candidate: PromptCandidate) -> None:
        candidate.island_id = self.island_id
        self.population.append(candidate)
        if len(self.population) > self.archive_size:
            # Evict lowest combined_score
            self.population.sort(key=lambda c: c.combined_score, reverse=True)
            self.population = self.population[:self.archive_size]
        if (self.best_candidate is None or
            candidate.combined_score > self.best_candidate.combined_score):
            self.best_candidate = candidate

@dataclass(frozen=True)
class BehavioralScoreBreakdown:
    """Detailed behavioral score per the paper's four components."""
    syntax_validity: float      # Weight: 0.35
    entry_point_defined: float  # Weight: 0.35
    risk_free_patterns: float   # Weight: 0.20
    no_undesirable: float       # Weight: 0.10
    total: float                # Weighted sum, clamped [0, 1]

@dataclass
class CalibrationResult:
    """Output of MAP-RPE for a single model on a single task."""
    model_id: str
    task_id: str
    optimal_prompt: str
    performance_score: float
    behavioral_score: float
    combined_score: float
    iterations_used: int
    total_api_calls: int
    total_tokens: int
    duration_seconds: float
    all_candidates: list[PromptCandidate] = field(default_factory=list)

@dataclass
class CalibrationPair:
    """Source-target calibrated prompt pair for one alignment task."""
    task_id: str
    source_result: CalibrationResult
    target_result: CalibrationResult
    task_info: str  # Description/metadata about the alignment task

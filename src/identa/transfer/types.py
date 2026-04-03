from dataclasses import dataclass, field
from datetime import datetime

@dataclass(frozen=True)
class TransferableKnowledge:
    source_model: str
    target_model: str
    summary: str              
    num_alignment_tasks: int  
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    calibration_hash: str = ""  

    def cache_key(self) -> str:
        return f"{self.source_model}__{self.target_model}__{self.calibration_hash}"

@dataclass
class TransferResult:
    original_prompt_id: str
    source_prompt_text: str
    transferred_prompt_text: str
    source_model: str
    target_model: str
    knowledge_used: str       
    adapter_model: str        
    latency_ms: float
    tokens_used: int

@dataclass
class MigrationReport:
    source_model: str
    target_model: str
    total_prompts: int
    successful: int
    failed: int
    total_duration_seconds: float
    total_tokens: int
    estimated_cost_usd: float
    results: list[TransferResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    knowledge: TransferableKnowledge | None = None

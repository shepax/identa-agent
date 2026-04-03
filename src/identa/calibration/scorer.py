import ast
import re
from identa.calibration.types import BehavioralScoreBreakdown
from identa.config.settings import ScorerConfig
from identa.tasks.schema import TaskInstance

# Patterns considered unsafe
RISK_PATTERNS = [
    r'\bexec\s*\(', r'\beval\s*\(', r'\bos\.system\s*\(',
    r'\bsubprocess\b', r'\b__import__\b', r'\bopen\s*\(',
    r'\bos\.remove\b', r'\bshutil\.rmtree\b', r'\bos\.unlink\b',
]

# Patterns considered undesirable
UNDESIRABLE_PATTERNS = [
    r'\bprint\s*\((?!.*["\'](?:Error|Warning|Exception))',
    r'#\s*TODO', r'#\s*FIXME', r'#\s*HACK',
    r'= (?:0|1|42|100|999)\b(?!\.)',
]

class BehavioralScorer:
    def __init__(self, config: ScorerConfig, domain: str = "coding"):
        self.config = config
        self.domain = domain
        # Apply domain overrides if present
        overrides = config.domain_overrides.get(domain, {})
        self.w_syntax = overrides.get("syntax", config.syntax_validity_weight)
        self.w_entry = overrides.get("entry_point", config.entry_point_weight)
        self.w_risk = overrides.get("risk_free", config.risk_free_weight)
        self.w_undesirable = overrides.get("no_undesirable", config.no_undesirable_weight)

    def score(self, output: str, instance: TaskInstance) -> BehavioralScoreBreakdown:
        syntax = self._check_syntax(output, instance)
        entry_point = self._check_entry_point(output, instance)
        risk_free = self._check_risk_free(output)
        no_undesirable = self._check_no_undesirable(output)

        total = (
            self.w_syntax * syntax +
            self.w_entry * entry_point +
            self.w_risk * risk_free +
            self.w_undesirable * no_undesirable
        )
        total = max(0.0, min(1.0, total))

        return BehavioralScoreBreakdown(
            syntax_validity=syntax,
            entry_point_defined=entry_point,
            risk_free_patterns=risk_free,
            no_undesirable=no_undesirable,
            total=total,
        )

    def _check_syntax(self, output: str, instance: TaskInstance) -> float:
        if instance.metadata.get("type") == "conceptual":
            return 1.0
            
        code = self._extract_code(output)
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0

    def _check_entry_point(self, output: str, instance: TaskInstance) -> float:
        if instance.metadata.get("type") == "conceptual":
            return 1.0
            
        entry_point = instance.metadata.get("entry_point")
        if not entry_point:
            return 1.0
        code = self._extract_code(output)
        pattern = rf'def\s+{re.escape(entry_point)}\s*\('
        return 1.0 if re.search(pattern, code) else 0.0

    def _check_risk_free(self, output: str) -> float:
        code = self._extract_code(output)
        violations = sum(1 for p in RISK_PATTERNS if re.search(p, code))
        if violations == 0:
            return 1.0
        return max(0.0, 1.0 - (violations * 0.25))

    def _check_no_undesirable(self, output: str) -> float:
        code = self._extract_code(output)
        violations = sum(1 for p in UNDESIRABLE_PATTERNS if re.search(p, code))
        if violations == 0:
            return 1.0
        return max(0.0, 1.0 - (violations * 0.2))

    def _extract_code(self, output: str) -> str:
        match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
        if match:
            return match.group(1)
        return output

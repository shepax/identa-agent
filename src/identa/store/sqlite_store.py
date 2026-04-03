import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from identa.store.models import Base, PromptRecord, CalibrationRecord, MigrationRecord

logger = logging.getLogger(__name__)

# AUDIT-FIX: 3.4 — generous but bounded limit to prevent OOM from adversarial inputs
MAX_PROMPT_LENGTH = 100_000


class SqliteStore:
    """Persistent storage for prompts and calibration results using SQLite."""

    def __init__(self, db_path: str = "~/.identa/store.db"):
        expanded_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)

        self.engine = create_engine(f"sqlite:///{expanded_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_prompt(self, id: str, content: str, model_id: str, format: str = "raw_text") -> None:
        """Persist a prompt record.

        AUDIT-FIX: 3.4 — Validates ID non-empty and content within size bound.
        """
        # AUDIT-FIX: 3.4 — Input validation
        if not id or not id.strip():
            raise ValueError("Prompt ID cannot be empty")
        if len(content) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt content of {len(content)} chars exceeds max {MAX_PROMPT_LENGTH}"
            )

        with self.Session() as session:
            record = PromptRecord(
                id=id,
                content=content,
                source_model=model_id,
                format=format
            )
            session.merge(record)
            session.commit()

    def get_prompt(self, id: str) -> str | None:
        """Retrieve a prompt by ID."""
        with self.Session() as session:
            record = session.query(PromptRecord).filter_by(id=id).first()
            return record.content if record else None

    def save_calibration(self, result) -> None:
        """Persist a calibration result."""
        with self.Session() as session:
            record = CalibrationRecord(
                model_id=result.model_id,
                task_id=result.task_id,
                optimal_prompt=result.optimal_prompt,
                performance_score=result.performance_score,
                behavioral_score=result.behavioral_score,
                combined_score=result.combined_score
            )
            session.add(record)
            session.commit()

    def save_calibration_with_prompt(
        self,
        prompt_id: str,
        prompt_content: str,
        model_id: str,
        result,
    ) -> None:
        """Atomic write: prompt + calibration record in a single transaction.

        AUDIT-FIX: 1.2 — Prevents inconsistent state if the process crashes
        between writing the prompt and writing the calibration record.
        """
        # Validate before opening the transaction
        if not prompt_id or not prompt_id.strip():
            raise ValueError("Prompt ID cannot be empty")
        if len(prompt_content) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt content of {len(prompt_content)} chars exceeds max {MAX_PROMPT_LENGTH}"
            )

        with self.Session() as session:
            try:
                prompt = PromptRecord(
                    id=prompt_id,
                    content=prompt_content,
                    source_model=model_id,
                )
                session.merge(prompt)  # upsert prompt

                cal = CalibrationRecord(
                    model_id=result.model_id,
                    task_id=result.task_id,
                    optimal_prompt=result.optimal_prompt,
                    performance_score=result.performance_score,
                    behavioral_score=result.behavioral_score,
                    combined_score=result.combined_score,
                )
                session.add(cal)
                session.commit()  # AUDIT-FIX: 1.2 — both writes committed atomically
            except Exception:
                session.rollback()
                logger.error(
                    f"Failed atomic write for prompt {prompt_id!r} / "
                    f"model {model_id!r}. Rolled back."
                )
                raise

    def get(self, key: str):
        """Cache-compatible get: model_id:task_id"""
        if ":" not in key:
            return None
        mid, tid = key.split(":", 1)
        with self.Session() as session:
            record = session.query(CalibrationRecord).filter_by(
                model_id=mid, task_id=tid
            ).order_by(CalibrationRecord.created_at.desc()).first()
            if not record:
                return None

            from identa.calibration.types import CalibrationResult
            return CalibrationResult(
                model_id=record.model_id,
                task_id=record.task_id,
                optimal_prompt=record.optimal_prompt,
                performance_score=record.performance_score,
                behavioral_score=record.behavioral_score,
                combined_score=record.combined_score,
                iterations_used=0,
                total_api_calls=0,
                total_tokens=0,
                duration_seconds=0.0
            )

    def put(self, key: str, result) -> None:
        """Cache-compatible put."""
        self.save_calibration(result)

    def list_prompts(self):
        """Return all prompt records."""
        with self.Session() as session:
            return session.query(PromptRecord).all()

    def save_migration(
        self,
        source_model: str,
        target_model: str,
        source_prompt_id: str,
        target_prompt_content: str,
        transfer_gap: float = 0.0
    ) -> None:
        """Persist a migration record."""
        with self.Session() as session:
            record = MigrationRecord(
                source_model=source_model,
                target_model=target_model,
                source_prompt_id=source_prompt_id,
                target_prompt_content=target_prompt_content,
                transfer_gap=transfer_gap
            )
            session.add(record)
            session.commit()

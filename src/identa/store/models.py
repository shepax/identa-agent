from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PromptRecord(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True)
    name = Column(String)
    content = Column(String, nullable=False)
    format = Column(String)  # raw_text, json_messages, etc.
    source_model = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class CalibrationRecord(Base):
    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, nullable=False)
    task_id = Column(String, nullable=False)
    optimal_prompt = Column(String, nullable=False)
    performance_score = Column(Float)
    behavioral_score = Column(Float)
    combined_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # AUDIT-FIX: 1.1 — Add indexes on hot query paths (.get() filters by model_id+task_id)
    __table_args__ = (
        Index("ix_calibrations_model_task", "model_id", "task_id"),
        Index("ix_calibrations_created", "created_at"),
    )


class MigrationRecord(Base):
    __tablename__ = "migrations"

    id = Column(Integer, primary_key=True)
    source_model = Column(String, nullable=False)
    target_model = Column(String, nullable=False)
    source_prompt_id = Column(String)
    target_prompt_content = Column(String, nullable=False)
    transfer_gap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # AUDIT-FIX: 1.1 — Index on source+target for migration lookup queries
    __table_args__ = (
        Index("ix_migrations_models", "source_model", "target_model"),
    )

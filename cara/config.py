"""
CARA-X Configuration
====================
Centralized configuration with Pydantic Settings.
All values come from environment variables or .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global CARA-X configuration. Every field maps to an env variable."""

    # --- Project paths ---
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent
    )

    # --- LLM Provider (Groq) ---
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use for reasoning",
    )

    # --- Neo4j (Optional — disabled by default) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "cara_x_2026"
    neo4j_enabled: bool = False

    # --- Qdrant (Optional — disabled by default) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_enabled: bool = False

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    log_level: str = "INFO"

    # --- CARA-X Engine ---
    consolidation_interval_seconds: int = 300
    consolidation_confidence_threshold: float = 0.7
    consolidation_min_occurrences: int = 3
    episodic_decay_hours: int = 72
    max_episodic_memories: int = 10000
    metacognition_tracking: bool = True

    # --- Safety ---
    sandbox_mode: bool = True
    human_in_the_loop: bool = True
    confidence_threshold_for_action: float = 0.8

    # --- Storage paths ---
    @property
    def data_dir(self) -> Path:
        p = self.project_root / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def procedural_db_path(self) -> Path:
        return self.data_dir / "procedural_memory.db"

    @property
    def audit_log_path(self) -> Path:
        return self.data_dir / "audit_log.jsonl"

    @property
    def episodic_snapshot_path(self) -> Path:
        return self.data_dir / "episodic_snapshot.json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

"""
L4: Procedural Memory
======================
Stores learned action sequences (procedures/skills) with success tracking.
Uses in-memory dict with optional SQLite persistence.
"""
from __future__ import annotations
import json, sqlite3, time, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from loguru import logger

@dataclass
class Procedure:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    preconditions: dict[str, Any] = field(default_factory=dict)
    environment: str = ""
    total_executions: int = 0
    successful_executions: int = 0
    last_executed: float = 0.0
    avg_reward: float = 0.0
    created_at: float = field(default_factory=time.time)
    source: str = "learned"
    version: int = 1

    @property
    def success_rate(self) -> float:
        return self.successful_executions / self.total_executions if self.total_executions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "goal": self.goal,
            "steps": self.steps, "preconditions": self.preconditions,
            "environment": self.environment,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": self.success_rate, "avg_reward": self.avg_reward,
            "source": self.source, "version": self.version,
        }

class ProceduralMemory:
    def __init__(self, db_path: Path | None = None):
        self._procedures: dict[str, Procedure] = {}
        self._db_path = db_path
        if db_path:
            self._init_db()
            self._load_from_db()
        logger.info("ProceduralMemory initialized ({} procedures)", len(self._procedures))

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""CREATE TABLE IF NOT EXISTS procedures (
            id TEXT PRIMARY KEY, name TEXT, goal TEXT, data TEXT,
            success_rate REAL, total_executions INTEGER, created_at REAL)""")
        conn.commit(); conn.close()

    def _load_from_db(self):
        if not self._db_path or not self._db_path.exists(): return
        try:
            conn = sqlite3.connect(str(self._db_path))
            for row in conn.execute("SELECT id, data FROM procedures"):
                data = json.loads(row[1])
                proc = Procedure(**{k: v for k, v in data.items() if k in Procedure.__dataclass_fields__})
                self._procedures[proc.id] = proc
            conn.close()
        except Exception as e:
            logger.warning("Failed to load procedures: {}", e)

    def _save_to_db(self, proc: Procedure):
        if not self._db_path: return
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("INSERT OR REPLACE INTO procedures VALUES (?,?,?,?,?,?,?)",
                (proc.id, proc.name, proc.goal, json.dumps(proc.to_dict(), default=str),
                 proc.success_rate, proc.total_executions, proc.created_at))
            conn.commit(); conn.close()
        except Exception as e:
            logger.warning("Failed to save procedure: {}", e)

    def add_procedure(self, name: str, goal: str, steps: list[dict[str, Any]],
                      preconditions: dict[str, Any] | None = None,
                      environment: str = "", source: str = "learned") -> Procedure:
        proc = Procedure(name=name, goal=goal, steps=steps,
                        preconditions=preconditions or {}, environment=environment, source=source)
        self._procedures[proc.id] = proc
        self._save_to_db(proc)
        logger.info("Added procedure: '{}' ({} steps)", name, len(steps))
        return proc

    def record_execution(self, procedure_id: str, success: bool, reward: float = 0.0):
        proc = self._procedures.get(procedure_id)
        if not proc: return
        proc.total_executions += 1
        if success: proc.successful_executions += 1
        proc.last_executed = time.time()
        proc.avg_reward = (proc.avg_reward * (proc.total_executions - 1) + reward) / proc.total_executions
        self._save_to_db(proc)

    def find_procedure(self, goal: str | None = None, environment: str | None = None,
                       min_success_rate: float = 0.0) -> list[Procedure]:
        results = []
        for proc in self._procedures.values():
            if goal and goal.lower() not in proc.goal.lower(): continue
            if environment and proc.environment and proc.environment != environment: continue
            if proc.success_rate < min_success_rate: continue
            results.append(proc)
        return sorted(results, key=lambda p: p.success_rate, reverse=True)

    def get_all(self) -> list[Procedure]:
        return list(self._procedures.values())

    def get_stats(self) -> dict[str, Any]:
        procs = list(self._procedures.values())
        if not procs: return {"total_procedures": 0}
        return {
            "total_procedures": len(procs),
            "avg_success_rate": sum(p.success_rate for p in procs) / len(procs),
            "total_executions": sum(p.total_executions for p in procs),
        }

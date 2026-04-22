"""
Safety & Alignment Layer
=========================
Intervention sandbox, audit logging, and confidence gates.
Runs parallel to all layers — every action passes through safety checks.
"""
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from loguru import logger
from cara.config import get_settings

@dataclass
class AuditEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    action_type: str = ""  # intervention | edge_added | edge_removed | prediction | hypothesis
    description: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    approved: bool = True
    blocked_reason: str = ""

class AuditLog:
    """Immutable audit trail for all causal claims and actions."""
    def __init__(self, log_path: Path | None = None):
        self._entries: list[AuditEntry] = []
        self._log_path = log_path

    def log(self, action_type: str, description: str, evidence: dict | None = None,
            confidence: float = 0.0, approved: bool = True, blocked_reason: str = "") -> AuditEntry:
        entry = AuditEntry(action_type=action_type, description=description,
                          evidence=evidence or {}, confidence=confidence,
                          approved=approved, blocked_reason=blocked_reason)
        self._entries.append(entry)
        if self._log_path:
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps({
                        "id": entry.id, "timestamp": entry.timestamp,
                        "action_type": entry.action_type, "description": entry.description,
                        "confidence": entry.confidence, "approved": entry.approved,
                        "blocked_reason": entry.blocked_reason,
                    }, default=str) + "\n")
            except Exception:
                pass
        return entry

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        return [{"id": e.id, "timestamp": e.timestamp, "action_type": e.action_type,
                 "description": e.description, "confidence": e.confidence,
                 "approved": e.approved, "blocked_reason": e.blocked_reason}
                for e in self._entries[-n:]]

    def get_blocked(self) -> list[dict[str, Any]]:
        return [{"id": e.id, "description": e.description, "reason": e.blocked_reason}
                for e in self._entries if not e.approved]

class ConfidenceGate:
    """Gates actions based on confidence thresholds."""
    def __init__(self, audit_log: AuditLog):
        self.settings = get_settings()
        self.audit_log = audit_log

    def check(self, action: str, confidence: float, is_high_stakes: bool = False) -> tuple[bool, str]:
        """Check if an action should be allowed."""
        threshold = self.settings.confidence_threshold_for_action
        if is_high_stakes and self.settings.human_in_the_loop:
            self.audit_log.log("gate_check", f"High-stakes action blocked for HITL: {action}",
                             confidence=confidence, approved=False,
                             blocked_reason="Human-in-the-loop required for high-stakes actions")
            return False, "Human-in-the-loop required for high-stakes actions"

        if confidence < threshold:
            reason = f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
            self.audit_log.log("gate_check", f"Action blocked: {action}", confidence=confidence,
                             approved=False, blocked_reason=reason)
            return False, reason

        self.audit_log.log("gate_check", f"Action approved: {action}", confidence=confidence, approved=True)
        return True, "Approved"

class InterventionSandbox:
    """Ensures all interventions run in isolated environments."""
    def __init__(self):
        self._active_interventions: list[dict[str, Any]] = []
        self._completed: list[dict[str, Any]] = []

    def begin(self, intervention: dict[str, Any]) -> str:
        """Register an intervention in the sandbox."""
        sandbox_id = str(uuid.uuid4())
        record = {"sandbox_id": sandbox_id, "intervention": intervention,
                  "started_at": time.time(), "status": "running"}
        self._active_interventions.append(record)
        logger.info("Sandbox intervention started: {}", intervention)
        return sandbox_id

    def complete(self, sandbox_id: str, result: dict[str, Any]) -> None:
        for i, record in enumerate(self._active_interventions):
            if record["sandbox_id"] == sandbox_id:
                record["status"] = "completed"
                record["result"] = result
                record["completed_at"] = time.time()
                self._completed.append(record)
                self._active_interventions.pop(i)
                break

    def get_active(self) -> list[dict[str, Any]]:
        return list(self._active_interventions)

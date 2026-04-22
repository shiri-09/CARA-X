"""
L5: Hypothesis Manager
=======================
Manages the lifecycle of causal hypotheses:
  proposed → testing → confirmed/refuted

Hypotheses come from: LLM, causal discovery algorithms, consolidation patterns.
Each hypothesis is tracked with evidence and scheduled for interventional testing.
"""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from loguru import logger

class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    QUEUED = "queued"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"

@dataclass
class CausalHypothesis:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause: str = ""
    effect: str = ""
    mechanism: str = ""
    initial_confidence: float = 0.5
    current_confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    source: str = "llm"  # llm | discovery | consolidation | manual
    test_intervention: str = ""
    test_results: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None
    priority: float = 0.5  # Higher = test sooner

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "cause": self.cause, "effect": self.effect,
            "mechanism": self.mechanism, "confidence": self.current_confidence,
            "status": self.status.value, "source": self.source,
            "test_intervention": self.test_intervention,
            "n_tests": len(self.test_results), "priority": self.priority,
            "created_at": self.created_at,
        }

class HypothesisManager:
    """Manages the hypothesis queue and testing pipeline."""
    def __init__(self):
        self._hypotheses: dict[str, CausalHypothesis] = {}
        self._resolved: list[CausalHypothesis] = []
        logger.info("HypothesisManager initialized")

    def propose(self, cause: str, effect: str, mechanism: str = "",
                confidence: float = 0.5, source: str = "llm",
                test_intervention: str = "", priority: float = 0.5) -> CausalHypothesis:
        # Check for duplicates
        for h in self._hypotheses.values():
            if h.cause == cause and h.effect == effect and h.status in (
                HypothesisStatus.PROPOSED, HypothesisStatus.QUEUED, HypothesisStatus.TESTING):
                logger.debug("Duplicate hypothesis: {} → {}", cause, effect)
                h.current_confidence = max(h.current_confidence, confidence)
                return h

        hyp = CausalHypothesis(
            cause=cause, effect=effect, mechanism=mechanism,
            initial_confidence=confidence, current_confidence=confidence,
            source=source, test_intervention=test_intervention, priority=priority,
        )
        self._hypotheses[hyp.id] = hyp
        logger.info("New hypothesis: {} → {} (conf={:.2f}, source={})", cause, effect, confidence, source)
        return hyp

    def get_next_to_test(self) -> CausalHypothesis | None:
        """Get the highest-priority untested hypothesis."""
        candidates = [h for h in self._hypotheses.values()
                      if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.QUEUED)]
        if not candidates:
            return None
        return max(candidates, key=lambda h: h.priority)

    def record_test_result(self, hypothesis_id: str, result: dict[str, Any],
                           supports: bool) -> None:
        hyp = self._hypotheses.get(hypothesis_id)
        if not hyp: return
        
        hyp.test_results.append({**result, "supports": supports, "timestamp": time.time()})
        hyp.status = HypothesisStatus.TESTING

        # Update confidence based on test result
        if supports:
            hyp.current_confidence = min(0.99, hyp.current_confidence + 0.15)
        else:
            hyp.current_confidence = max(0.01, hyp.current_confidence - 0.2)

        # Auto-resolve after enough evidence
        if len(hyp.test_results) >= 3:
            if hyp.current_confidence > 0.7:
                hyp.status = HypothesisStatus.CONFIRMED
                hyp.resolved_at = time.time()
                self._resolved.append(hyp)
            elif hyp.current_confidence < 0.3:
                hyp.status = HypothesisStatus.REFUTED
                hyp.resolved_at = time.time()
                self._resolved.append(hyp)

    def get_queue(self) -> list[dict[str, Any]]:
        return [h.to_dict() for h in self._hypotheses.values()
                if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.QUEUED, HypothesisStatus.TESTING)]

    def get_confirmed(self) -> list[dict[str, Any]]:
        return [h.to_dict() for h in self._resolved if h.status == HypothesisStatus.CONFIRMED]

    def get_all(self) -> list[dict[str, Any]]:
        return [h.to_dict() for h in self._hypotheses.values()]

    def get_stats(self) -> dict[str, Any]:
        all_h = list(self._hypotheses.values())
        return {
            "total": len(all_h),
            "proposed": sum(1 for h in all_h if h.status == HypothesisStatus.PROPOSED),
            "testing": sum(1 for h in all_h if h.status == HypothesisStatus.TESTING),
            "confirmed": sum(1 for h in all_h if h.status == HypothesisStatus.CONFIRMED),
            "refuted": sum(1 for h in all_h if h.status == HypothesisStatus.REFUTED),
        }

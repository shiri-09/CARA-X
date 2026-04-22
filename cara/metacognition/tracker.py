"""
L6: Metacognition — Prediction Tracker
========================================
Logs every prediction + actual outcome. Computes running accuracy
per domain/subgraph. Detects concept drift (accuracy degradation).
"""
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from loguru import logger

@dataclass
class PredictionRecord:
    id: int = 0
    timestamp: float = field(default_factory=time.time)
    target_variable: str = ""
    predicted_value: float = 0.0
    predicted_ci_low: float = 0.0
    predicted_ci_high: float = 0.0
    actual_value: float = 0.0
    error: float = 0.0
    within_ci: bool = False
    domain: str = ""

class PredictionTracker:
    """Tracks prediction accuracy over time for metacognitive monitoring."""
    def __init__(self):
        self._records: list[PredictionRecord] = []
        self._by_variable: dict[str, list[PredictionRecord]] = defaultdict(list)
        self._running_accuracy: float = 0.0
        self._window_size = 50
        logger.info("PredictionTracker initialized")

    def record(self, target: str, predicted: float, actual: float,
               ci_low: float = 0.0, ci_high: float = 0.0, domain: str = "") -> PredictionRecord:
        rec = PredictionRecord(
            id=len(self._records),
            target_variable=target,
            predicted_value=predicted,
            predicted_ci_low=ci_low, predicted_ci_high=ci_high,
            actual_value=actual,
            error=abs(predicted - actual),
            within_ci=(ci_low <= actual <= ci_high) if ci_low != ci_high else False,
            domain=domain,
        )
        self._records.append(rec)
        self._by_variable[target].append(rec)
        return rec

    def get_accuracy(self, variable: str | None = None, window: int | None = None) -> dict[str, float]:
        """Get prediction accuracy metrics."""
        records = self._by_variable.get(variable, []) if variable else self._records
        if window:
            records = records[-window:]
        if not records:
            return {"mae": 0, "rmse": 0, "ci_coverage": 0, "n_predictions": 0}

        errors = [r.error for r in records]
        ci_hits = [r.within_ci for r in records]
        return {
            "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
            "ci_coverage": float(np.mean(ci_hits)),
            "n_predictions": len(records),
            "trend": self._compute_trend(errors),
        }

    def detect_concept_drift(self, window: int = 20) -> dict[str, Any]:
        """Detect if prediction accuracy is degrading (concept drift)."""
        if len(self._records) < window * 2:
            return {"drift_detected": False, "reason": "Insufficient data"}

        old_errors = [r.error for r in self._records[-window*2:-window]]
        new_errors = [r.error for r in self._records[-window:]]
        old_mean = np.mean(old_errors)
        new_mean = np.mean(new_errors)

        drift = new_mean > old_mean * 1.5
        return {
            "drift_detected": drift,
            "old_mae": float(old_mean),
            "new_mae": float(new_mean),
            "degradation_pct": float((new_mean - old_mean) / (old_mean + 1e-8) * 100),
        }

    def _compute_trend(self, errors: list[float]) -> str:
        if len(errors) < 10:
            return "insufficient_data"
        recent = np.mean(errors[-5:])
        older = np.mean(errors[-10:-5])
        if recent < older * 0.8:
            return "improving"
        elif recent > older * 1.2:
            return "degrading"
        return "stable"

    def get_learning_curve(self) -> list[dict[str, Any]]:
        """Get data points for the learning curve visualization."""
        if not self._records:
            return []
        points = []
        window = max(1, len(self._records) // 50)
        for i in range(window, len(self._records), window):
            chunk = self._records[i-window:i]
            errors = [r.error for r in chunk]
            points.append({
                "step": i, "mae": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(np.array(errors)**2))),
                "ci_coverage": float(np.mean([r.within_ci for r in chunk])),
                "timestamp": chunk[-1].timestamp,
            })
        return points

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_predictions": len(self._records),
            "variables_tracked": len(self._by_variable),
            **self.get_accuracy(),
            "drift": self.detect_concept_drift(),
        }

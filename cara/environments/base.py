"""
L1: Environment Interface — Abstract Base
==========================================

Defines the contract that ALL environments must implement.
Every environment produces (state, action, outcome, timestamp) tuples
that feed into the causal discovery engine.

The abstraction supports:
  - Observational data collection (passive)
  - Interventional experiments (active do-calculus)
  - Ground-truth causal graph (for benchmarking)
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ActionType(Enum):
    """Whether an action is observational or interventional (do-operator)."""
    OBSERVE = "observe"
    INTERVENE = "intervene"  # do(X = x) — forces a variable to a value


@dataclass
class Observation:
    """
    A single (state, action, outcome, timestamp) tuple.
    This is the atomic unit of data flowing through CARA-X.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Pre-action state (dict of variable_name → value)
    state: dict[str, Any] = field(default_factory=dict)

    # Action taken
    action: str = ""
    action_type: ActionType = ActionType.OBSERVE
    action_params: dict[str, Any] = field(default_factory=dict)

    # Post-action state
    outcome: dict[str, Any] = field(default_factory=dict)

    # Optional reward signal
    reward: float = 0.0

    # Context metadata
    environment: str = ""
    episode: int = 0
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "state": self.state,
            "action": self.action,
            "action_type": self.action_type.value,
            "action_params": self.action_params,
            "outcome": self.outcome,
            "reward": self.reward,
            "environment": self.environment,
            "episode": self.episode,
            "step": self.step,
        }


@dataclass
class CausalEdge:
    """Represents a ground-truth causal relationship for benchmarking."""
    cause: str
    effect: str
    effect_size: float = 1.0
    delay: float = 0.0
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentInfo:
    """Metadata about an environment."""
    name: str
    description: str
    variables: list[str]
    actions: list[str]
    ground_truth_edges: list[CausalEdge] = field(default_factory=list)


class CausalEnvironment(ABC):
    """
    Abstract base class for all CARA-X environments.

    Environments are the data source for causal discovery.
    They must support:
      1. Reset to initial state
      2. Step with an action (observe or intervene)
      3. Provide variable names and possible actions
      4. Optionally provide ground-truth causal graph (for evaluation)
    """

    def __init__(self, name: str):
        self.name = name
        self._episode = 0
        self._step = 0
        self._history: list[Observation] = []

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset environment, return initial state dict."""
        ...

    @abstractmethod
    def step(self, action: str, params: dict[str, Any] | None = None) -> Observation:
        """Execute an action and return an Observation."""
        ...

    @abstractmethod
    def intervene(self, variable: str, value: Any) -> Observation:
        """
        Execute a do-operator intervention: do(variable = value).
        Forces a variable to a specific value and observes the outcome.
        This breaks the natural data-generating process (removes incoming edges).
        """
        ...

    @abstractmethod
    def get_info(self) -> EnvironmentInfo:
        """Return environment metadata including variable names and actions."""
        ...

    def get_ground_truth(self) -> list[CausalEdge] | None:
        """
        Return ground-truth causal edges if available.
        Used for benchmarking causal discovery accuracy.
        Returns None if no ground truth is available.
        """
        return None

    def get_history(self) -> list[Observation]:
        """Return all collected observations."""
        return list(self._history)

    def get_data_matrix(self) -> tuple[np.ndarray, list[str]]:
        """
        Convert collected observations to a data matrix for causal discovery.
        Returns: (data_matrix, column_names)
        """
        if not self._history:
            return np.array([]), []

        # Collect all variable names from outcomes
        all_vars = set()
        for obs in self._history:
            all_vars.update(obs.outcome.keys())
        var_names = sorted(all_vars)

        # Build matrix
        rows = []
        for obs in self._history:
            row = [obs.outcome.get(v, np.nan) for v in var_names]
            rows.append(row)

        return np.array(rows, dtype=float), var_names

    def _record(self, obs: Observation) -> None:
        """Record an observation to history."""
        obs.environment = self.name
        obs.episode = self._episode
        obs.step = self._step
        self._history.append(obs)
        self._step += 1

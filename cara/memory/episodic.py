"""
L4: Episodic Memory
====================

Fast, high-fidelity, decaying memory for raw experiences.

Stores (state, action, outcome, timestamp) tuples with full context.
Supports:
  - Vector similarity search (for retrieving relevant past experiences)
  - Exponential decay (old memories fade)
  - Pruning after consolidation
  - In-memory backend with Qdrant-ready abstraction

Inspired by hippocampal fast learning in CLS theory.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from loguru import logger


@dataclass
class EpisodicMemory:
    """A single episodic memory entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # The raw experience
    state: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    action_type: str = "observe"  # observe | intervene
    outcome: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0

    # Context
    environment: str = ""
    episode: int = 0
    step: int = 0

    # Memory management
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    consolidated: bool = False  # Has this been promoted to semantic memory?
    importance: float = 0.5  # Computed from surprise/novelty

    # Embedding for similarity search
    embedding: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "state": self.state,
            "action": self.action,
            "action_type": self.action_type,
            "outcome": self.outcome,
            "reward": self.reward,
            "environment": self.environment,
            "episode": self.episode,
            "step": self.step,
            "access_count": self.access_count,
            "consolidated": self.consolidated,
            "importance": self.importance,
            "age_hours": (time.time() - self.timestamp) / 3600,
        }


class EpisodicMemoryBackend(Protocol):
    """Protocol for swappable backends (in-memory ↔ Qdrant)."""
    def store(self, memory: EpisodicMemory) -> None: ...
    def search_similar(self, query_embedding: np.ndarray, k: int) -> list[EpisodicMemory]: ...
    def get_recent(self, n: int) -> list[EpisodicMemory]: ...
    def get_by_id(self, memory_id: str) -> EpisodicMemory | None: ...
    def get_unconsolidated(self) -> list[EpisodicMemory]: ...
    def mark_consolidated(self, memory_id: str) -> None: ...
    def prune_old(self, max_age_hours: float) -> int: ...
    def count(self) -> int: ...


class InMemoryEpisodicBackend:
    """
    In-memory episodic memory store using numpy for similarity search.
    Easily replaceable with Qdrant for production.
    """

    def __init__(self, max_size: int = 10000):
        self._memories: dict[str, EpisodicMemory] = {}
        self._max_size = max_size

    def store(self, memory: EpisodicMemory) -> None:
        """Store a new episodic memory."""
        # Auto-prune if at capacity
        if len(self._memories) >= self._max_size:
            self._evict_oldest()

        # Compute embedding if not provided
        if memory.embedding is None:
            memory.embedding = self._compute_embedding(memory)

        self._memories[memory.id] = memory

    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> list[EpisodicMemory]:
        """Find k most similar memories by cosine similarity."""
        if not self._memories:
            return []

        memories = list(self._memories.values())
        embeddings = []
        valid_memories = []

        for m in memories:
            if m.embedding is not None:
                embeddings.append(m.embedding)
                valid_memories.append(m)

        if not embeddings:
            return []

        emb_matrix = np.vstack(embeddings)
        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
        similarities = (emb_matrix / norms) @ query_norm

        # Sort by similarity (descending)
        indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in indices:
            mem = valid_memories[idx]
            mem.access_count += 1
            mem.last_accessed = time.time()
            results.append(mem)

        return results

    def get_recent(self, n: int = 20) -> list[EpisodicMemory]:
        """Get the N most recent memories."""
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.timestamp,
            reverse=True,
        )
        return sorted_memories[:n]

    def get_by_id(self, memory_id: str) -> EpisodicMemory | None:
        return self._memories.get(memory_id)

    def get_unconsolidated(self) -> list[EpisodicMemory]:
        """Get all memories not yet promoted to semantic memory."""
        return [m for m in self._memories.values() if not m.consolidated]

    def mark_consolidated(self, memory_id: str) -> None:
        if memory_id in self._memories:
            self._memories[memory_id].consolidated = True

    def prune_old(self, max_age_hours: float = 72.0) -> int:
        """Remove consolidated memories older than max_age_hours."""
        cutoff = time.time() - max_age_hours * 3600
        to_remove = [
            mid for mid, m in self._memories.items()
            if m.consolidated and m.timestamp < cutoff
        ]
        for mid in to_remove:
            del self._memories[mid]
        return len(to_remove)

    def prune_to_size(self, target_size: int) -> int:
        """Prune to target size, removing oldest consolidated memories first."""
        if len(self._memories) <= target_size:
            return 0

        # Sort: consolidated first, then by age (oldest first)
        sorted_mems = sorted(
            self._memories.items(),
            key=lambda x: (not x[1].consolidated, x[1].timestamp),
        )

        n_to_remove = len(self._memories) - target_size
        removed = 0
        for mid, _ in sorted_mems[:n_to_remove]:
            del self._memories[mid]
            removed += 1

        return removed

    def count(self) -> int:
        return len(self._memories)

    def get_all(self) -> list[EpisodicMemory]:
        return list(self._memories.values())

    def _evict_oldest(self):
        """Remove the oldest consolidated memory, or oldest overall."""
        consolidated = [(mid, m) for mid, m in self._memories.items() if m.consolidated]
        if consolidated:
            oldest = min(consolidated, key=lambda x: x[1].timestamp)
            del self._memories[oldest[0]]
        else:
            oldest = min(self._memories.items(), key=lambda x: x[1].timestamp)
            del self._memories[oldest[0]]

    def _compute_embedding(self, memory: EpisodicMemory) -> np.ndarray:
        """
        Compute a simple embedding from the memory's state/outcome values.
        In production, this would use a proper embedding model.
        """
        values = []
        for v in memory.state.values():
            if isinstance(v, (int, float)):
                values.append(float(v))
        for v in memory.outcome.values():
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            return np.zeros(64)

        # Pad or truncate to fixed size
        arr = np.array(values[:64], dtype=float)
        if len(arr) < 64:
            arr = np.pad(arr, (0, 64 - len(arr)))

        # Normalize
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        return arr


class EpisodicStore:
    """
    High-level episodic memory manager.
    Wraps the backend with additional logic for importance scoring
    and memory statistics.
    """

    def __init__(self, backend: InMemoryEpisodicBackend | None = None, max_size: int = 10000):
        self.backend = backend or InMemoryEpisodicBackend(max_size=max_size)
        self._running_reward_mean = 0.0
        self._running_reward_var = 1.0
        self._n_stored = 0
        logger.info("EpisodicStore initialized (max_size={})", max_size)

    def store_observation(self, observation: Any) -> EpisodicMemory:
        """
        Store an observation from the environment as an episodic memory.
        Computes importance based on surprise (deviation from expected reward).
        """
        from cara.environments.base import Observation

        if isinstance(observation, Observation):
            memory = EpisodicMemory(
                state=observation.state,
                action=observation.action,
                action_type=observation.action_type.value,
                outcome=observation.outcome,
                reward=observation.reward,
                environment=observation.environment,
                episode=observation.episode,
                step=observation.step,
                importance=self._compute_importance(observation.reward),
            )
        else:
            memory = EpisodicMemory(
                state=observation.get("state", {}),
                action=observation.get("action", ""),
                outcome=observation.get("outcome", {}),
                reward=observation.get("reward", 0.0),
            )

        self.backend.store(memory)
        self._n_stored += 1
        return memory

    def recall_similar(self, query_state: dict[str, Any], k: int = 5) -> list[EpisodicMemory]:
        """Find memories similar to the given state."""
        # Convert state to embedding
        values = [float(v) for v in query_state.values() if isinstance(v, (int, float))]
        arr = np.array(values[:64], dtype=float)
        if len(arr) < 64:
            arr = np.pad(arr, (0, 64 - len(arr)))
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        return self.backend.search_similar(arr, k=k)

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        all_mems = self.backend.get_all() if hasattr(self.backend, 'get_all') else []
        consolidated = sum(1 for m in all_mems if m.consolidated)

        return {
            "total_memories": self.backend.count(),
            "total_stored": self._n_stored,
            "consolidated": consolidated,
            "unconsolidated": self.backend.count() - consolidated,
        }

    def _compute_importance(self, reward: float) -> float:
        """
        Compute importance based on surprise (large deviation from running mean).
        High-surprise events are remembered more strongly.
        """
        self._n_stored += 1
        # Welford's online algorithm
        delta = reward - self._running_reward_mean
        self._running_reward_mean += delta / max(self._n_stored, 1)
        delta2 = reward - self._running_reward_mean
        self._running_reward_var += (delta * delta2 - self._running_reward_var) / max(self._n_stored, 1)

        std = max(self._running_reward_var ** 0.5, 0.01)
        surprise = abs(reward - self._running_reward_mean) / std
        # Sigmoid to [0, 1]
        importance = 1.0 / (1.0 + np.exp(-surprise + 1))
        return float(importance)

"""
L3: Dynamic Causal World Model
================================

The central causal DAG that represents CARA-X's understanding of the world.

In-memory implementation using NetworkX with a clean abstraction layer
that can be swapped to Neo4j for production deployment.

Each edge carries rich metadata:
  - effect_size: How strong is A→B?
  - confidence: Bayesian posterior probability
  - temporal_delay: A causes B after ~Δt seconds
  - conditions: A→B only when C is true
  - evidence_count: N observations supporting/refuting
  - last_verified: When was this edge last confirmed?
  - evidence_type: 'observational' | 'interventional' | 'llm_hypothesis'

Each node carries:
  - observable_range: (min, max) of observed values
  - intervention_history: List of past do() operations
  - uncertainty_level: Bayesian uncertainty over this node's role
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import networkx as nx
import numpy as np
from loguru import logger


class EvidenceType(Enum):
    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"
    LLM_HYPOTHESIS = "llm_hypothesis"


@dataclass
class EdgeMetadata:
    """Rich metadata attached to each causal edge."""
    effect_size: float = 0.0
    confidence: float = 0.5
    temporal_delay: float = 0.0
    conditions: dict[str, Any] = field(default_factory=dict)
    evidence_count: int = 0
    evidence_for: int = 0
    evidence_against: int = 0
    last_verified: float = field(default_factory=time.time)
    evidence_type: EvidenceType = EvidenceType.OBSERVATIONAL
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_size": self.effect_size,
            "confidence": self.confidence,
            "temporal_delay": self.temporal_delay,
            "conditions": self.conditions,
            "evidence_count": self.evidence_count,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "last_verified": self.last_verified,
            "evidence_type": self.evidence_type.value,
            "created_at": self.created_at,
        }


@dataclass
class NodeMetadata:
    """Rich metadata attached to each causal node."""
    observable_min: float = float("inf")
    observable_max: float = float("-inf")
    mean_value: float = 0.0
    std_value: float = 0.0
    observation_count: int = 0
    intervention_history: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_level: float = 1.0  # 1.0 = total uncertainty, 0.0 = certain
    last_observed: float = field(default_factory=time.time)

    def update_statistics(self, value: float):
        """Online update of running statistics."""
        self.observation_count += 1
        self.observable_min = min(self.observable_min, value)
        self.observable_max = max(self.observable_max, value)
        # Welford's online algorithm for mean and variance
        delta = value - self.mean_value
        self.mean_value += delta / self.observation_count
        delta2 = value - self.mean_value
        self._m2 = getattr(self, "_m2", 0.0) + delta * delta2
        self.std_value = (self._m2 / self.observation_count) ** 0.5 if self.observation_count > 1 else 0.0
        self.last_observed = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_range": [self.observable_min, self.observable_max],
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "observation_count": self.observation_count,
            "uncertainty_level": self.uncertainty_level,
            "intervention_count": len(self.intervention_history),
            "last_observed": self.last_observed,
        }


class WorldModelBackend(Protocol):
    """Protocol for swappable graph backends (in-memory ↔ Neo4j)."""

    def add_node(self, name: str, metadata: NodeMetadata) -> None: ...
    def add_edge(self, cause: str, effect: str, metadata: EdgeMetadata) -> None: ...
    def remove_edge(self, cause: str, effect: str) -> None: ...
    def get_nodes(self) -> list[str]: ...
    def get_edges(self) -> list[tuple[str, str, EdgeMetadata]]: ...
    def get_edge_metadata(self, cause: str, effect: str) -> EdgeMetadata | None: ...
    def get_node_metadata(self, name: str) -> NodeMetadata | None: ...
    def get_parents(self, node: str) -> list[str]: ...
    def get_children(self, node: str) -> list[str]: ...
    def has_edge(self, cause: str, effect: str) -> bool: ...
    def get_networkx_graph(self) -> nx.DiGraph: ...
    def clear(self) -> None: ...


class InMemoryGraphBackend:
    """
    NetworkX-based in-memory graph backend.
    Stores full EdgeMetadata and NodeMetadata on the graph objects.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_meta: dict[str, NodeMetadata] = {}
        self._edge_meta: dict[tuple[str, str], EdgeMetadata] = {}

    def add_node(self, name: str, metadata: NodeMetadata | None = None) -> None:
        if metadata is None:
            metadata = NodeMetadata()
        self._graph.add_node(name)
        self._node_meta[name] = metadata

    def add_edge(self, cause: str, effect: str, metadata: EdgeMetadata | None = None) -> None:
        if metadata is None:
            metadata = EdgeMetadata()
        # Ensure nodes exist
        if cause not in self._graph:
            self.add_node(cause)
        if effect not in self._graph:
            self.add_node(effect)
        self._graph.add_edge(cause, effect)
        self._edge_meta[(cause, effect)] = metadata

    def remove_edge(self, cause: str, effect: str) -> None:
        if self._graph.has_edge(cause, effect):
            self._graph.remove_edge(cause, effect)
            self._edge_meta.pop((cause, effect), None)

    def get_nodes(self) -> list[str]:
        return list(self._graph.nodes())

    def get_edges(self) -> list[tuple[str, str, EdgeMetadata]]:
        result = []
        for u, v in self._graph.edges():
            meta = self._edge_meta.get((u, v), EdgeMetadata())
            result.append((u, v, meta))
        return result

    def get_edge_metadata(self, cause: str, effect: str) -> EdgeMetadata | None:
        return self._edge_meta.get((cause, effect))

    def get_node_metadata(self, name: str) -> NodeMetadata | None:
        return self._node_meta.get(name)

    def get_parents(self, node: str) -> list[str]:
        if node not in self._graph:
            return []
        return list(self._graph.predecessors(node))

    def get_children(self, node: str) -> list[str]:
        if node not in self._graph:
            return []
        return list(self._graph.successors(node))

    def has_edge(self, cause: str, effect: str) -> bool:
        return self._graph.has_edge(cause, effect)

    def get_networkx_graph(self) -> nx.DiGraph:
        return self._graph.copy()

    def clear(self) -> None:
        self._graph.clear()
        self._node_meta.clear()
        self._edge_meta.clear()

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()


class CausalWorldModel:
    """
    L3: The Dynamic Causal World Model.

    This is the brain of CARA-X — a living, evolving causal graph
    that represents the system's understanding of cause and effect.

    Supports:
      - Adding/updating/removing causal edges with rich metadata
      - Bayesian confidence updating
      - Monte Carlo prediction rollouts
      - d-separation queries
      - Graph serialization for API/frontend
      - Backend swapping (in-memory ↔ Neo4j)
    """

    def __init__(self, backend: InMemoryGraphBackend | None = None):
        self.backend = backend or InMemoryGraphBackend()
        self._prediction_log: list[dict[str, Any]] = []
        logger.info("CausalWorldModel initialized with {} backend",
                     type(self.backend).__name__)

    # ----------------------------------------------------------------
    # Core graph operations
    # ----------------------------------------------------------------

    def add_causal_edge(
        self,
        cause: str,
        effect: str,
        effect_size: float = 0.0,
        confidence: float = 0.5,
        evidence_type: EvidenceType = EvidenceType.OBSERVATIONAL,
        temporal_delay: float = 0.0,
        conditions: dict[str, Any] | None = None,
    ) -> EdgeMetadata:
        """
        Add or update a causal edge A → B.

        If the edge already exists, Bayesian-update the confidence
        rather than overwriting it.
        """
        existing = self.backend.get_edge_metadata(cause, effect)

        if existing is not None:
            # Bayesian update: combine old and new evidence
            meta = self._bayesian_update_edge(existing, confidence, evidence_type)
            meta.effect_size = (existing.effect_size * 0.7 + effect_size * 0.3)
            meta.temporal_delay = temporal_delay or existing.temporal_delay
            if conditions:
                meta.conditions.update(conditions)
            meta.last_verified = time.time()
        else:
            meta = EdgeMetadata(
                effect_size=effect_size,
                confidence=confidence,
                temporal_delay=temporal_delay,
                conditions=conditions or {},
                evidence_count=1,
                evidence_for=1,
                evidence_type=evidence_type,
            )

        self.backend.add_edge(cause, effect, meta)

        # Check for cycles — causal graphs must be DAGs
        if not nx.is_directed_acyclic_graph(self.backend.get_networkx_graph()):
            logger.warning("Cycle detected! Removing edge {} → {}", cause, effect)
            self.backend.remove_edge(cause, effect)
            return existing or meta

        logger.debug("Causal edge: {} → {} (conf={:.3f}, type={})",
                      cause, effect, meta.confidence, evidence_type.value)
        return meta

    def remove_causal_edge(self, cause: str, effect: str) -> None:
        """Remove a causal edge (refuted by evidence)."""
        self.backend.remove_edge(cause, effect)
        logger.info("Removed causal edge: {} → {}", cause, effect)

    def update_node_observation(self, node: str, value: float) -> None:
        """Update running statistics for a node based on observed value."""
        meta = self.backend.get_node_metadata(node)
        if meta is None:
            meta = NodeMetadata()
            self.backend.add_node(node, meta)
        meta.update_statistics(value)

    def record_intervention(self, node: str, value: Any, outcome: dict[str, Any]) -> None:
        """Record that an intervention was performed on this node."""
        meta = self.backend.get_node_metadata(node)
        if meta is None:
            meta = NodeMetadata()
            self.backend.add_node(node, meta)
        meta.intervention_history.append({
            "value": value,
            "outcome": outcome,
            "timestamp": time.time(),
        })

    # ----------------------------------------------------------------
    # Bayesian confidence updating
    # ----------------------------------------------------------------

    def _bayesian_update_edge(
        self,
        existing: EdgeMetadata,
        new_confidence: float,
        evidence_type: EvidenceType,
    ) -> EdgeMetadata:
        """
        Bayesian update of edge confidence.
        Interventional evidence is weighted more heavily than observational.
        """
        # Weight: interventional evidence counts 3x observational
        weight = 3.0 if evidence_type == EvidenceType.INTERVENTIONAL else 1.0
        if evidence_type == EvidenceType.LLM_HYPOTHESIS:
            weight = 0.5  # LLM hypotheses get low initial weight

        existing.evidence_count += 1
        if new_confidence > 0.5:
            existing.evidence_for += 1
        else:
            existing.evidence_against += 1

        # Weighted running average
        total_weight = existing.evidence_count + weight
        existing.confidence = (
            existing.confidence * existing.evidence_count + new_confidence * weight
        ) / total_weight

        existing.confidence = max(0.01, min(0.99, existing.confidence))
        return existing

    # ----------------------------------------------------------------
    # Monte Carlo simulation / prediction
    # ----------------------------------------------------------------

    def predict(
        self,
        intervention: dict[str, float],
        target_nodes: list[str] | None = None,
        n_samples: int = 1000,
    ) -> dict[str, dict[str, float]]:
        """
        Monte Carlo rollout: "If I do(X=x), predict Y ± CI."

        Args:
            intervention: {variable: value} — the do() operation
            target_nodes: Which nodes to predict (None = all descendants)
            n_samples: Number of Monte Carlo samples

        Returns:
            {node: {"mean": ..., "std": ..., "ci_low": ..., "ci_high": ...}}
        """
        graph = self.backend.get_networkx_graph()
        if not graph.nodes():
            return {}

        # Find all downstream nodes from intervention points
        downstream = set()
        for node in intervention:
            if node in graph:
                downstream.update(nx.descendants(graph, node))

        if target_nodes:
            downstream = downstream.intersection(target_nodes)

        if not downstream:
            return {}

        # Topological order for simulation
        try:
            topo_order = [n for n in nx.topological_sort(graph) if n in downstream]
        except nx.NetworkXUnfeasible:
            logger.warning("Graph has cycles, cannot perform Monte Carlo rollout")
            return {}

        # Run Monte Carlo samples
        samples: dict[str, list[float]] = {n: [] for n in downstream}

        for _ in range(n_samples):
            node_values: dict[str, float] = dict(intervention)

            for node in topo_order:
                if node in intervention:
                    continue  # Interventioned nodes are fixed

                parents = self.backend.get_parents(node)
                if not parents:
                    node_meta = self.backend.get_node_metadata(node)
                    if node_meta and node_meta.observation_count > 0:
                        val = np.random.normal(node_meta.mean_value, node_meta.std_value)
                    else:
                        val = 0.0
                    node_values[node] = val
                    samples[node].append(val)
                    continue

                # Compute value from parent effects
                val = 0.0
                for parent in parents:
                    edge_meta = self.backend.get_edge_metadata(parent, node)
                    parent_val = node_values.get(parent, 0.0)
                    if edge_meta:
                        effect = edge_meta.effect_size * parent_val
                        # Add noise proportional to uncertainty
                        noise = np.random.normal(0, (1 - edge_meta.confidence) * abs(effect) + 0.1)
                        val += effect + noise

                node_values[node] = val
                samples[node].append(val)

        # Compute statistics
        results = {}
        for node, vals in samples.items():
            if vals:
                arr = np.array(vals)
                results[node] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "ci_low": float(np.percentile(arr, 2.5)),
                    "ci_high": float(np.percentile(arr, 97.5)),
                }

        return results

    # ----------------------------------------------------------------
    # Graph queries
    # ----------------------------------------------------------------

    def d_separated(self, x: str, y: str, z: set[str] | None = None) -> bool:
        """
        Test if X and Y are d-separated given Z.
        Uses NetworkX's d-separation implementation.
        """
        graph = self.backend.get_networkx_graph()
        z = z or set()
        try:
            return nx.d_separated(graph, {x}, {y}, z)
        except Exception:
            return False

    def get_markov_blanket(self, node: str) -> set[str]:
        """Get the Markov blanket of a node (parents + children + co-parents)."""
        graph = self.backend.get_networkx_graph()
        if node not in graph:
            return set()

        blanket = set()
        # Parents
        blanket.update(graph.predecessors(node))
        # Children
        children = list(graph.successors(node))
        blanket.update(children)
        # Co-parents (other parents of children)
        for child in children:
            blanket.update(graph.predecessors(child))
        blanket.discard(node)
        return blanket

    def get_causal_path(self, source: str, target: str) -> list[str] | None:
        """Find a causal path from source to target."""
        graph = self.backend.get_networkx_graph()
        try:
            return nx.shortest_path(graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_causal_paths(self, source: str, target: str) -> list[list[str]]:
        """Find all causal paths from source to target."""
        graph = self.backend.get_networkx_graph()
        try:
            return list(nx.all_simple_paths(graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_root_causes(self, node: str) -> list[str]:
        """Find all root causes (ancestors with no parents) of a node."""
        graph = self.backend.get_networkx_graph()
        if node not in graph:
            return []
        ancestors = nx.ancestors(graph, node)
        return [a for a in ancestors if graph.in_degree(a) == 0]

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire world model for API/frontend."""
        nodes = []
        for name in self.backend.get_nodes():
            meta = self.backend.get_node_metadata(name)
            nodes.append({
                "id": name,
                "metadata": meta.to_dict() if meta else {},
            })

        edges = []
        for cause, effect, meta in self.backend.get_edges():
            edges.append({
                "source": cause,
                "target": effect,
                "metadata": meta.to_dict(),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "is_dag": nx.is_directed_acyclic_graph(self.backend.get_networkx_graph()),
        }

    def save_snapshot(self, path: Path) -> None:
        """Save graph snapshot to JSON file."""
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved world model snapshot to {}", path)

    def get_graph_metrics(self) -> dict[str, Any]:
        """Compute graph-level metrics."""
        graph = self.backend.get_networkx_graph()
        edges = self.backend.get_edges()

        confidences = [m.confidence for _, _, m in edges] if edges else [0]
        evidence_counts = [m.evidence_count for _, _, m in edges] if edges else [0]

        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "density": nx.density(graph) if graph.nodes() else 0,
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "avg_evidence_count": float(np.mean(evidence_counts)),
            "total_evidence": int(np.sum(evidence_counts)),
            "connected_components": nx.number_weakly_connected_components(graph) if graph.nodes() else 0,
        }

    # ----------------------------------------------------------------
    # Comparison with ground truth (for benchmarking)
    # ----------------------------------------------------------------

    def compare_with_ground_truth(
        self, ground_truth_edges: list[tuple[str, str]]
    ) -> dict[str, float]:
        """
        Compare discovered graph against ground truth.
        Returns SHD, F1, precision, recall.
        """
        discovered = set((u, v) for u, v in self.backend.get_networkx_graph().edges())
        truth = set(ground_truth_edges)

        tp = len(discovered & truth)
        fp = len(discovered - truth)
        fn = len(truth - discovered)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = fp + fn  # Structural Hamming Distance

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "shd": shd,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

"""
L4: Semantic Memory
====================

Interface to the Causal World Model (L3) as semantic memory.

In CLS theory, semantic memory represents slow-learned, compressed,
generalized knowledge — in CARA-X, this IS the causal graph.

Causal rules extracted from episodic replay are stored as edges
in the graph with Bayesian-updated confidence.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from cara.core.world_model import CausalWorldModel, EvidenceType


class SemanticMemory:
    """
    Semantic memory = the causal graph.
    Provides a memory-centric interface to the world model.
    """

    def __init__(self, world_model: CausalWorldModel):
        self.world_model = world_model
        logger.info("SemanticMemory initialized (backed by CausalWorldModel)")

    def store_causal_rule(
        self,
        cause: str,
        effect: str,
        confidence: float,
        effect_size: float = 0.0,
        evidence_type: EvidenceType = EvidenceType.OBSERVATIONAL,
        conditions: dict[str, Any] | None = None,
    ) -> None:
        """
        Store (or update) a causal rule in semantic memory.
        This is the promotion target for episodic consolidation.
        """
        self.world_model.add_causal_edge(
            cause=cause,
            effect=effect,
            effect_size=effect_size,
            confidence=confidence,
            evidence_type=evidence_type,
            conditions=conditions,
        )
        logger.debug("Stored causal rule: {} → {} (conf={:.3f})", cause, effect, confidence)

    def query_causes(self, variable: str) -> list[dict[str, Any]]:
        """What causes this variable? Returns parent nodes with edge metadata."""
        parents = self.world_model.backend.get_parents(variable)
        results = []
        for parent in parents:
            meta = self.world_model.backend.get_edge_metadata(parent, variable)
            results.append({
                "cause": parent,
                "effect": variable,
                "confidence": meta.confidence if meta else 0.0,
                "effect_size": meta.effect_size if meta else 0.0,
                "evidence_count": meta.evidence_count if meta else 0,
            })
        return results

    def query_effects(self, variable: str) -> list[dict[str, Any]]:
        """What does this variable cause? Returns child nodes with edge metadata."""
        children = self.world_model.backend.get_children(variable)
        results = []
        for child in children:
            meta = self.world_model.backend.get_edge_metadata(variable, child)
            results.append({
                "cause": variable,
                "effect": child,
                "confidence": meta.confidence if meta else 0.0,
                "effect_size": meta.effect_size if meta else 0.0,
                "evidence_count": meta.evidence_count if meta else 0,
            })
        return results

    def explain_relationship(self, source: str, target: str) -> dict[str, Any]:
        """
        Explain the causal relationship between two variables.
        Traces all paths and returns evidence.
        """
        paths = self.world_model.get_all_causal_paths(source, target)

        if not paths:
            return {
                "source": source,
                "target": target,
                "relationship": "no_causal_path",
                "paths": [],
            }

        path_details = []
        for path in paths:
            edges = []
            total_confidence = 1.0
            for i in range(len(path) - 1):
                meta = self.world_model.backend.get_edge_metadata(path[i], path[i + 1])
                if meta:
                    edges.append({
                        "from": path[i],
                        "to": path[i + 1],
                        "confidence": meta.confidence,
                        "effect_size": meta.effect_size,
                        "evidence_type": meta.evidence_type.value,
                    })
                    total_confidence *= meta.confidence

            path_details.append({
                "path": path,
                "length": len(path) - 1,
                "total_confidence": total_confidence,
                "edges": edges,
            })

        return {
            "source": source,
            "target": target,
            "relationship": "causal",
            "n_paths": len(paths),
            "paths": sorted(path_details, key=lambda p: p["total_confidence"], reverse=True),
        }

    def get_knowledge_summary(self) -> dict[str, Any]:
        """Summary of all semantic knowledge."""
        metrics = self.world_model.get_graph_metrics()
        edges = self.world_model.backend.get_edges()

        high_conf = [e for e in edges if e[2].confidence > 0.8]
        low_conf = [e for e in edges if e[2].confidence < 0.3]
        interventional = [e for e in edges if e[2].evidence_type == EvidenceType.INTERVENTIONAL]

        return {
            **metrics,
            "high_confidence_edges": len(high_conf),
            "low_confidence_edges": len(low_conf),
            "interventionally_verified": len(interventional),
        }

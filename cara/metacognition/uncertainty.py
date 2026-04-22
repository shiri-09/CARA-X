"""
L6: Metacognition — Uncertainty Mapping & Explorer
====================================================
Bayesian uncertainty over graph structure.
Identifies low-evidence regions and designs experiments
to maximally reduce uncertainty (expected information gain).
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from loguru import logger
from cara.core.world_model import CausalWorldModel

class UncertaintyMapper:
    """Maps uncertainty across the causal graph for metacognitive awareness."""
    def __init__(self, world_model: CausalWorldModel):
        self.world_model = world_model

    def get_uncertainty_map(self) -> dict[str, Any]:
        """Compute uncertainty for every node and edge in the graph."""
        edges = self.world_model.backend.get_edges()
        nodes = self.world_model.backend.get_nodes()

        edge_uncertainties = []
        for cause, effect, meta in edges:
            uncertainty = 1.0 - meta.confidence
            evidence_factor = 1.0 / (1.0 + meta.evidence_count * 0.1)
            combined = uncertainty * 0.7 + evidence_factor * 0.3
            edge_uncertainties.append({
                "source": cause, "target": effect,
                "uncertainty": float(combined),
                "confidence": meta.confidence,
                "evidence_count": meta.evidence_count,
                "evidence_type": meta.evidence_type.value,
            })

        node_uncertainties = []
        for node in nodes:
            meta = self.world_model.backend.get_node_metadata(node)
            if meta:
                node_uncertainties.append({
                    "node": node,
                    "uncertainty": meta.uncertainty_level,
                    "observation_count": meta.observation_count,
                    "intervention_count": len(meta.intervention_history),
                })

        # Identify high-uncertainty regions
        high_uncertainty_edges = [e for e in edge_uncertainties if e["uncertainty"] > 0.5]
        low_evidence_nodes = [n for n in node_uncertainties if n["observation_count"] < 10]

        return {
            "edges": edge_uncertainties,
            "nodes": node_uncertainties,
            "high_uncertainty_edges": len(high_uncertainty_edges),
            "low_evidence_nodes": len(low_evidence_nodes),
            "avg_edge_uncertainty": float(np.mean([e["uncertainty"] for e in edge_uncertainties])) if edge_uncertainties else 1.0,
            "avg_node_uncertainty": float(np.mean([n["uncertainty"] for n in node_uncertainties])) if node_uncertainties else 1.0,
        }


class CuriosityExplorer:
    """
    Curiosity-driven exploration: selects the next intervention that
    maximally reduces uncertainty (expected information gain).
    """
    def __init__(self, world_model: CausalWorldModel, uncertainty_mapper: UncertaintyMapper):
        self.world_model = world_model
        self.mapper = uncertainty_mapper

    def suggest_next_intervention(self) -> dict[str, Any] | None:
        """Suggest the best intervention to reduce global uncertainty."""
        uncertainty_map = self.mapper.get_uncertainty_map()
        
        if not uncertainty_map["edges"]:
            # No graph yet — suggest exploring any variable
            nodes = self.world_model.backend.get_nodes()
            if nodes:
                return {"variable": nodes[0], "value": 0, "reason": "No graph structure yet — explore any variable", "expected_info_gain": 1.0}
            return None

        # Rank edges by uncertainty
        candidates = sorted(uncertainty_map["edges"], key=lambda e: e["uncertainty"], reverse=True)
        
        if not candidates:
            return None

        best = candidates[0]
        # Compute expected information gain (entropy reduction)
        # H(edge) = -p*log(p) - (1-p)*log(1-p), where p = confidence
        p = max(0.01, min(0.99, best["confidence"]))
        current_entropy = -p * math.log2(p) - (1-p) * math.log2(1-p)
        # After intervention, we expect to learn the direction — entropy drops
        expected_post_entropy = current_entropy * 0.3  # Heuristic
        info_gain = current_entropy - expected_post_entropy

        return {
            "variable": best["source"],
            "target_edge": f"{best['source']} → {best['target']}",
            "current_confidence": best["confidence"],
            "current_uncertainty": best["uncertainty"],
            "expected_info_gain": float(info_gain),
            "reason": f"Edge {best['source']}→{best['target']} has high uncertainty ({best['uncertainty']:.2f}). Intervening on {best['source']} will clarify this relationship.",
        }

    def get_exploration_priorities(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Rank all possible interventions by expected information gain."""
        uncertainty_map = self.mapper.get_uncertainty_map()
        priorities = []

        for edge in uncertainty_map["edges"]:
            p = max(0.01, min(0.99, edge["confidence"]))
            entropy = -p * math.log2(p) - (1-p) * math.log2(1-p)
            priorities.append({
                "edge": f"{edge['source']} → {edge['target']}",
                "intervene_on": edge["source"],
                "uncertainty": edge["uncertainty"],
                "entropy": float(entropy),
                "evidence_count": edge["evidence_count"],
                "priority_score": float(entropy * (1 + 1/(edge["evidence_count"]+1))),
            })

        return sorted(priorities, key=lambda p: p["priority_score"], reverse=True)[:top_k]


class CompetenceBoundary:
    """Determines what the system knows vs doesn't know."""
    def __init__(self, world_model: CausalWorldModel, uncertainty_mapper: UncertaintyMapper):
        self.world_model = world_model
        self.mapper = uncertainty_mapper

    def assess(self) -> dict[str, Any]:
        """Assess competence boundaries."""
        edges = self.world_model.backend.get_edges()
        high_conf = [(c, e, m) for c, e, m in edges if m.confidence > 0.8]
        med_conf = [(c, e, m) for c, e, m in edges if 0.4 <= m.confidence <= 0.8]
        low_conf = [(c, e, m) for c, e, m in edges if m.confidence < 0.4]

        return {
            "confident_about": [{"edge": f"{c}→{e}", "confidence": m.confidence} for c, e, m in high_conf],
            "uncertain_about": [{"edge": f"{c}→{e}", "confidence": m.confidence} for c, e, m in med_conf],
            "dont_know": [{"edge": f"{c}→{e}", "confidence": m.confidence} for c, e, m in low_conf],
            "summary": {
                "high_confidence_edges": len(high_conf),
                "medium_confidence_edges": len(med_conf),
                "low_confidence_edges": len(low_conf),
                "total_edges": len(edges),
                "coverage_ratio": len(high_conf) / max(len(edges), 1),
            },
        }

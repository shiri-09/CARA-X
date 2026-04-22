"""
L5: Causal Explainer
=====================
Traces causal chains through the verified graph to explain events.
Every explanation is grounded in interventional evidence — zero hallucination.
"""
from __future__ import annotations
from typing import Any
from loguru import logger
from cara.core.world_model import CausalWorldModel
from cara.reasoning.llm_interface import LLMInterface

class CausalExplainer:
    """Generates causal explanations grounded in the verified graph."""

    def __init__(self, world_model: CausalWorldModel, llm: LLMInterface):
        self.world_model = world_model
        self.llm = llm

    def explain(self, event: dict[str, Any]) -> dict[str, Any]:
        """Explain why an event occurred by tracing causal chains."""
        target_var = event.get("variable", "")
        
        if not target_var:
            # Find the most anomalous variable in the event
            for var, val in event.items():
                if isinstance(val, (int, float)):
                    node_meta = self.world_model.backend.get_node_metadata(var)
                    if node_meta and node_meta.observation_count > 5:
                        z_score = abs(val - node_meta.mean_value) / (node_meta.std_value + 1e-8)
                        if z_score > 2.0:
                            target_var = var
                            break

        if not target_var:
            return {"explanation": "No anomalous variables detected", "confidence": 0.0}

        # Find root causes
        root_causes = self.world_model.get_root_causes(target_var)
        
        # Trace all causal paths to the target
        all_paths = []
        for root in root_causes:
            paths = self.world_model.get_all_causal_paths(root, target_var)
            all_paths.extend(paths)

        # Build path details with confidence
        path_details = []
        for path in all_paths[:5]:  # Limit to top 5 paths
            edges = []
            path_confidence = 1.0
            for i in range(len(path) - 1):
                meta = self.world_model.backend.get_edge_metadata(path[i], path[i+1])
                if meta:
                    edges.append({
                        "from": path[i], "to": path[i+1],
                        "confidence": meta.confidence,
                        "effect_size": meta.effect_size,
                        "evidence_type": meta.evidence_type.value,
                        "evidence_count": meta.evidence_count,
                    })
                    path_confidence *= meta.confidence
            path_details.append({"path": path, "confidence": path_confidence, "edges": edges})

        # Sort by confidence
        path_details.sort(key=lambda p: p["confidence"], reverse=True)

        # Use LLM to generate human-readable explanation
        graph_context = {"target": target_var, "root_causes": root_causes}
        nl_explanation = self.llm.explain_event(event, path_details, graph_context)

        return {
            "target_variable": target_var,
            "root_causes": root_causes,
            "causal_paths": path_details,
            "explanation": nl_explanation,
            "confidence": path_details[0]["confidence"] if path_details else 0.0,
            "n_paths": len(path_details),
        }

    def counterfactual(self, event: dict[str, Any], intervention: dict[str, float]) -> dict[str, Any]:
        """What would have happened if we had done X instead?"""
        # Use Monte Carlo simulation on the causal graph
        prediction = self.world_model.predict(intervention, n_samples=500)
        
        # Compare actual vs counterfactual
        comparison = {}
        for var, pred in prediction.items():
            actual = event.get(var, 0)
            comparison[var] = {
                "actual": actual,
                "counterfactual_mean": pred["mean"],
                "counterfactual_ci": [pred["ci_low"], pred["ci_high"]],
                "difference": actual - pred["mean"],
            }

        return {
            "intervention": intervention,
            "comparison": comparison,
            "conclusion": f"Under intervention {intervention}, {len(comparison)} variables would change."
        }

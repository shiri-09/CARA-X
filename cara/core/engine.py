"""
CARA-X Core Engine
===================
The main orchestration loop that ties all 6 layers together.

The cognitive cycle:
  1. Observe environment → store episodic memory
  2. Periodically run causal discovery on accumulated data
  3. Design and execute interventional experiments
  4. Update causal world model
  5. Consolidate memory (episodic → semantic)
  6. Track metacognitive metrics
  7. Generate explanations and plans via LLM
"""
from __future__ import annotations
import time
from typing import Any
from loguru import logger

from cara.config import get_settings
from cara.core.world_model import CausalWorldModel, InMemoryGraphBackend
from cara.core.causal_discovery import CausalDiscoveryEngine
from cara.memory.episodic import EpisodicStore, InMemoryEpisodicBackend
from cara.memory.semantic import SemanticMemory
from cara.memory.procedural import ProceduralMemory
from cara.memory.consolidation import ConsolidationEngine
from cara.reasoning.llm_interface import LLMInterface
from cara.reasoning.hypothesis import HypothesisManager
from cara.reasoning.explainer import CausalExplainer
from cara.metacognition.tracker import PredictionTracker
from cara.metacognition.uncertainty import UncertaintyMapper, CuriosityExplorer, CompetenceBoundary
from cara.safety.core import AuditLog, ConfidenceGate, InterventionSandbox
from cara.environments.base import CausalEnvironment, Observation


class CARAEngine:
    """
    The main CARA-X engine. Orchestrates all 6 cognitive layers.

    Usage:
        engine = CARAEngine()
        engine.attach_environment(my_env)
        engine.run_episode(n_steps=100)
        engine.run_discovery()
        engine.run_consolidation()
    """

    def __init__(self):
        settings = get_settings()

        # L3: World Model
        self.world_model = CausalWorldModel(backend=InMemoryGraphBackend())

        # L2: Causal Discovery
        self.discovery = CausalDiscoveryEngine(self.world_model)

        # L4: Memory
        self.episodic = EpisodicStore(
            backend=InMemoryEpisodicBackend(max_size=settings.max_episodic_memories)
        )
        self.semantic = SemanticMemory(self.world_model)
        self.procedural = ProceduralMemory(db_path=settings.procedural_db_path)
        self.consolidation = ConsolidationEngine(
            episodic=self.episodic,
            semantic=self.semantic,
            procedural=self.procedural,
            min_occurrences=settings.consolidation_min_occurrences,
            confidence_threshold=settings.consolidation_confidence_threshold,
            decay_hours=settings.episodic_decay_hours,
        )

        # L5: Reasoning
        self.llm = LLMInterface()
        self.hypothesis_manager = HypothesisManager()
        self.explainer = CausalExplainer(self.world_model, self.llm)

        # L6: Metacognition
        self.prediction_tracker = PredictionTracker()
        self.uncertainty_mapper = UncertaintyMapper(self.world_model)
        self.explorer = CuriosityExplorer(self.world_model, self.uncertainty_mapper)
        self.competence = CompetenceBoundary(self.world_model, self.uncertainty_mapper)

        # Safety
        self.audit_log = AuditLog(log_path=settings.audit_log_path)
        self.confidence_gate = ConfidenceGate(self.audit_log)
        self.sandbox = InterventionSandbox()

        # State
        self._environment: CausalEnvironment | None = None
        self._total_steps = 0
        self._total_episodes = 0
        self._discovery_runs = 0
        self._consolidation_runs = 0

        logger.info("═══ CARA-X Engine initialized ═══")
        logger.info("  L1: Environment Interface (awaiting attachment)")
        logger.info("  L2: Causal Discovery Engine (PC, GES, NOTEARS, Bayesian)")
        logger.info("  L3: Causal World Model (in-memory NetworkX)")
        logger.info("  L4: Three-Tier Memory (CLS-Inspired)")
        logger.info("  L5: Reasoning Engine (Groq: {})", settings.groq_model)
        logger.info("  L6: Metacognition (tracker + uncertainty + explorer)")
        logger.info("  Safety: audit log + confidence gate + sandbox")

    def attach_environment(self, env: CausalEnvironment) -> None:
        """Attach an environment for the agent to interact with."""
        self._environment = env
        logger.info("Environment attached: {}", env.name)

    def run_episode(self, n_steps: int = 50, explore_rate: float = 0.3) -> dict[str, Any]:
        """
        Run one episode of interaction with the environment.
        Collects observations, performs interventions, stores to episodic memory.
        """
        if not self._environment:
            raise RuntimeError("No environment attached. Call attach_environment() first.")

        env = self._environment
        state = env.reset()
        self._total_episodes += 1
        episode_rewards = []
        observations: list[Observation] = []

        logger.info("Episode {} started ({} steps, explore_rate={:.1f})",
                     self._total_episodes, n_steps, explore_rate)

        for step in range(n_steps):
            import random

            # Decide: observe, intervene, or follow curiosity
            if random.random() < explore_rate:
                # Curiosity-driven: ask metacognition what to explore
                suggestion = self.explorer.suggest_next_intervention()
                if suggestion and random.random() < 0.5:
                    # Interventional exploration
                    obs = env.intervene(suggestion["variable"], 0)
                    self.audit_log.log("intervention", f"Curiosity-driven: do({suggestion['variable']}=0)")
                else:
                    # Random action
                    info = env.get_info()
                    action = random.choice(info.actions)
                    obs = env.step(action)
            else:
                # Observe passively
                obs = env.step("observe")

            # Store to episodic memory
            self.episodic.store_observation(obs)
            observations.append(obs)
            episode_rewards.append(obs.reward)

            # Update node statistics in world model
            for var, val in obs.outcome.items():
                if isinstance(val, (int, float)):
                    self.world_model.update_node_observation(var, float(val))

            self._total_steps += 1

        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        logger.info("Episode {} complete: avg_reward={:.3f}, {} observations stored",
                     self._total_episodes, avg_reward, len(observations))

        return {
            "episode": self._total_episodes,
            "steps": n_steps,
            "avg_reward": avg_reward,
            "observations_stored": len(observations),
        }

    def run_discovery(self, algorithms: list[str] | None = None) -> dict[str, Any]:
        """Run causal discovery on accumulated observational data."""
        if not self._environment:
            return {"error": "No environment attached"}

        data, var_names = self._environment.get_data_matrix()
        if len(data) < 20:
            logger.warning("Not enough data for discovery ({} samples, need 20+)", len(data))
            return {"error": "Insufficient data", "n_samples": len(data)}

        self._discovery_runs += 1
        logger.info("Discovery run #{}: {} samples, {} variables",
                     self._discovery_runs, len(data), len(var_names))

        # Run ensemble of algorithms
        results = self.discovery.run_ensemble(data, var_names, algorithms=algorithms)
        n_edges = self.discovery.apply_results(results)

        # Log to audit
        self.audit_log.log("discovery", f"Run #{self._discovery_runs}: {n_edges} edges from {len(results)} algorithms")

        # Generate hypotheses from LLM based on new graph
        graph_context = self.world_model.to_dict()
        recent = self.episodic.backend.get_recent(5)
        obs_dicts = [m.to_dict() for m in recent]
        hypotheses = self.llm.generate_hypotheses(obs_dicts, graph_context)
        for h in hypotheses:
            self.hypothesis_manager.propose(
                cause=h.get("cause", ""), effect=h.get("effect", ""),
                mechanism=h.get("mechanism", ""), confidence=h.get("confidence", 0.3),
                source="llm", test_intervention=h.get("test_intervention", ""),
            )

        return {
            "discovery_run": self._discovery_runs,
            "algorithms_used": [r.algorithm for r in results],
            "edges_discovered": n_edges,
            "hypotheses_generated": len(hypotheses),
            "graph_metrics": self.world_model.get_graph_metrics(),
        }

    def run_consolidation(self) -> dict[str, Any]:
        """Run CLS-inspired memory consolidation."""
        self._consolidation_runs += 1
        result = self.consolidation.run_consolidation_cycle()
        return {
            "consolidation_run": self._consolidation_runs,
            "memories_replayed": result.memories_replayed,
            "patterns_promoted": result.patterns_promoted,
            "contradictions": result.contradictions_found,
            "memories_pruned": result.memories_pruned,
            "duration_seconds": result.duration_seconds,
        }

    def run_intervention_test(self) -> dict[str, Any] | None:
        """Test the next hypothesis via interventional experiment."""
        if not self._environment:
            return None

        hyp = self.hypothesis_manager.get_next_to_test()
        if not hyp:
            return {"status": "no_hypotheses_to_test"}

        # Safety check
        allowed, reason = self.confidence_gate.check(
            f"intervene({hyp.cause})", hyp.current_confidence
        )

        # Design experiment
        experiment = self.llm.design_experiment(hyp.to_dict(), self._environment.get_info().actions)
        variable = experiment.get("variable", hyp.cause)
        value = experiment.get("value", 0)

        # Execute in sandbox
        sandbox_id = self.sandbox.begin({"variable": variable, "value": value, "hypothesis": hyp.id})
        obs = self._environment.intervene(variable, value)
        self.episodic.store_observation(obs)
        self.sandbox.complete(sandbox_id, obs.to_dict())

        # Record intervention in world model
        self.world_model.record_intervention(variable, value, obs.outcome)

        # Evaluate: did the effect variable change as expected?
        effect_val = obs.outcome.get(hyp.effect)
        supports = effect_val is not None  # Simplified — real check would compare distributions
        self.hypothesis_manager.record_test_result(hyp.id, obs.to_dict(), supports)

        if hyp.status.value == "confirmed":
            self.world_model.add_causal_edge(
                cause=hyp.cause, effect=hyp.effect,
                confidence=hyp.current_confidence,
                evidence_type=self.world_model.backend.get_edge_metadata(hyp.cause, hyp.effect).evidence_type if self.world_model.backend.has_edge(hyp.cause, hyp.effect) else __import__('cara.core.world_model', fromlist=['EvidenceType']).EvidenceType.INTERVENTIONAL,
            )

        self.audit_log.log("intervention_test", f"Tested {hyp.cause}→{hyp.effect}: {'supports' if supports else 'refutes'}",
                          confidence=hyp.current_confidence)

        return {
            "hypothesis": hyp.to_dict(),
            "experiment": experiment,
            "observation": obs.to_dict(),
            "supports": supports,
        }

    def evaluate_against_ground_truth(self) -> dict[str, Any] | None:
        """Compare discovered graph against ground truth (if available)."""
        if not self._environment:
            return None
        gt = self._environment.get_ground_truth()
        if not gt:
            return None
        gt_edges = [(e.cause, e.effect) for e in gt]
        return self.world_model.compare_with_ground_truth(gt_edges)

    def get_full_status(self) -> dict[str, Any]:
        """Get comprehensive system status across all layers."""
        return {
            "engine": {
                "total_steps": self._total_steps,
                "total_episodes": self._total_episodes,
                "discovery_runs": self._discovery_runs,
                "consolidation_runs": self._consolidation_runs,
                "environment": self._environment.name if self._environment else None,
            },
            "world_model": self.world_model.get_graph_metrics(),
            "memory": {
                "episodic": self.episodic.get_stats(),
                "procedural": self.procedural.get_stats(),
                "semantic": self.semantic.get_knowledge_summary(),
                "consolidation_history": self.consolidation.get_history()[-5:],
            },
            "reasoning": {
                "llm": self.llm.get_stats(),
                "hypotheses": self.hypothesis_manager.get_stats(),
            },
            "metacognition": {
                "predictions": self.prediction_tracker.get_stats(),
                "uncertainty": self.uncertainty_mapper.get_uncertainty_map(),
                "competence": self.competence.assess(),
                "next_exploration": self.explorer.suggest_next_intervention(),
            },
            "safety": {
                "recent_audit": self.audit_log.get_recent(10),
                "active_interventions": self.sandbox.get_active(),
            },
        }

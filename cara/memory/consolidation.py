"""
L4: CLS-Inspired Memory Consolidation Engine
=============================================
Implements the Complementary Learning Systems theory:
  Raw experiences → Episodic buffer → Periodic replay →
  Extract causal regularities → Promote to semantic (causal graph) →
  Prune/archive consolidated episodes

Inspired by McClelland et al. (1995) and Kumaran et al. (2016).
"""
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from loguru import logger

from cara.memory.episodic import EpisodicStore, EpisodicMemory
from cara.memory.semantic import SemanticMemory
from cara.memory.procedural import ProceduralMemory
from cara.core.world_model import EvidenceType


@dataclass
class CausalPattern:
    """A candidate causal pattern extracted from episodic replay."""
    cause: str
    effect: str
    occurrences: int = 0
    correlation: float = 0.0
    avg_delay: float = 0.0
    confidence: float = 0.0
    effect_size: float = 0.0
    supporting_memories: list[str] = field(default_factory=list)


@dataclass
class ConsolidationResult:
    """Result of a single consolidation cycle."""
    timestamp: float = field(default_factory=time.time)
    memories_replayed: int = 0
    patterns_found: int = 0
    patterns_promoted: int = 0
    contradictions_found: int = 0
    memories_pruned: int = 0
    duration_seconds: float = 0.0


class ConsolidationEngine:
    """
    The CLS-inspired consolidation engine.
    
    Periodically:
      1. Sample from episodic buffer (replay)
      2. Extract causal regularities via correlation/co-occurrence
      3. If pattern appears N times with confidence > θ → promote to semantic
      4. Archive/prune consolidated episodic entries
      5. Flag contradictions for active investigation
    """
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        min_occurrences: int = 3,
        confidence_threshold: float = 0.7,
        decay_hours: float = 72.0,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold
        self.decay_hours = decay_hours
        self._history: list[ConsolidationResult] = []
        self._known_patterns: dict[tuple[str, str], CausalPattern] = {}
        logger.info("ConsolidationEngine initialized (min_occ={}, conf_θ={:.2f})",
                     min_occurrences, confidence_threshold)

    def run_consolidation_cycle(self) -> ConsolidationResult:
        """Execute one full consolidation cycle."""
        start = time.time()
        result = ConsolidationResult()

        # Step 1: Get unconsolidated memories
        memories = self.episodic.backend.get_unconsolidated()
        result.memories_replayed = len(memories)
        
        if len(memories) < 5:
            logger.debug("Too few memories for consolidation ({})", len(memories))
            return result

        logger.info("Running consolidation on {} episodic memories", len(memories))

        # Step 2: Extract causal patterns via co-occurrence analysis
        patterns = self._extract_patterns(memories)
        result.patterns_found = len(patterns)

        # Step 3: Promote high-confidence patterns to semantic memory
        promoted = 0
        contradictions = 0
        for pattern in patterns:
            if (pattern.occurrences >= self.min_occurrences and
                    pattern.confidence >= self.confidence_threshold):
                # Check for contradictions with existing knowledge
                existing = self.semantic.query_causes(pattern.effect)
                contradiction = False
                for ex in existing:
                    if ex["cause"] == pattern.cause and ex["confidence"] > 0.5:
                        if abs(ex["effect_size"] - pattern.effect_size) > 1.0:
                            contradictions += 1
                            contradiction = True
                            logger.warning("Contradiction: {} → {} (existing={:.2f}, new={:.2f})",
                                         pattern.cause, pattern.effect,
                                         ex["effect_size"], pattern.effect_size)

                if not contradiction:
                    self.semantic.store_causal_rule(
                        cause=pattern.cause,
                        effect=pattern.effect,
                        confidence=pattern.confidence,
                        effect_size=pattern.effect_size,
                        evidence_type=EvidenceType.OBSERVATIONAL,
                    )
                    promoted += 1
                    # Mark supporting memories as consolidated
                    for mem_id in pattern.supporting_memories:
                        self.episodic.backend.mark_consolidated(mem_id)

        result.patterns_promoted = promoted
        result.contradictions_found = contradictions

        # Step 4: Prune old consolidated memories
        result.memories_pruned = self.episodic.backend.prune_old(self.decay_hours)

        # Step 5: Extract action sequences for procedural memory
        self._extract_procedures(memories)

        result.duration_seconds = time.time() - start
        self._history.append(result)

        logger.info("Consolidation complete: {} patterns found, {} promoted, {} pruned in {:.2f}s",
                     result.patterns_found, result.patterns_promoted,
                     result.memories_pruned, result.duration_seconds)
        return result

    def _extract_patterns(self, memories: list[EpisodicMemory]) -> list[CausalPattern]:
        """Extract causal patterns from episodic memories via co-occurrence."""
        # Group memories by episode for sequential analysis
        episodes: dict[int, list[EpisodicMemory]] = defaultdict(list)
        for m in memories:
            episodes[m.episode].append(m)

        # Sort each episode by step
        for ep in episodes.values():
            ep.sort(key=lambda m: m.step)

        # Analyze variable co-changes across consecutive steps
        pair_stats: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)

        for ep_memories in episodes.values():
            for i in range(len(ep_memories) - 1):
                curr = ep_memories[i]
                nxt = ep_memories[i + 1]

                # Find variables that changed significantly
                for var_a in curr.outcome:
                    for var_b in nxt.outcome:
                        if var_a == var_b:
                            continue
                        val_a_before = curr.state.get(var_a, 0)
                        val_a_after = curr.outcome.get(var_a, 0)
                        val_b_before = nxt.state.get(var_b, 0)
                        val_b_after = nxt.outcome.get(var_b, 0)

                        try:
                            delta_a = float(val_a_after) - float(val_a_before)
                            delta_b = float(val_b_after) - float(val_b_before)
                        except (TypeError, ValueError):
                            continue

                        if abs(delta_a) > 0.1 and abs(delta_b) > 0.1:
                            pair_stats[(var_a, var_b)].append(
                                (delta_a, delta_b, curr.id)
                            )

        # Convert to CausalPattern objects
        patterns = []
        for (cause, effect), deltas in pair_stats.items():
            if len(deltas) < 2:
                continue

            delta_causes = [d[0] for d in deltas]
            delta_effects = [d[1] for d in deltas]
            mem_ids = [d[2] for d in deltas]

            # Compute correlation as confidence proxy
            if len(delta_causes) > 2:
                corr = np.corrcoef(delta_causes, delta_effects)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.5

            # Only keep positive correlations (cause and effect move together)
            if abs(corr) < 0.2:
                continue

            effect_size = np.mean(delta_effects) / (np.mean(np.abs(delta_causes)) + 1e-8)

            pattern = CausalPattern(
                cause=cause,
                effect=effect,
                occurrences=len(deltas),
                correlation=float(abs(corr)),
                confidence=float(min(0.95, abs(corr) * (1 + 0.1 * len(deltas)))),
                effect_size=float(effect_size),
                supporting_memories=mem_ids[:10],
            )
            patterns.append(pattern)

            # Track for future consolidation
            self._known_patterns[(cause, effect)] = pattern

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def _extract_procedures(self, memories: list[EpisodicMemory]):
        """Extract successful action sequences as procedures."""
        episodes: dict[int, list[EpisodicMemory]] = defaultdict(list)
        for m in memories:
            episodes[m.episode].append(m)

        for ep_memories in episodes.values():
            ep_memories.sort(key=lambda m: m.step)
            if len(ep_memories) < 3:
                continue

            # Look for sequences where reward improves
            if ep_memories[-1].reward > ep_memories[0].reward + 0.5:
                steps = [{"action": m.action, "params": getattr(m, 'action_params', {})}
                         for m in ep_memories if m.action and m.action != "observe"]
                if steps:
                    self.procedural.add_procedure(
                        name=f"auto_procedure_ep{ep_memories[0].episode}",
                        goal=f"Improve system state (reward +{ep_memories[-1].reward - ep_memories[0].reward:.2f})",
                        steps=steps,
                        environment=ep_memories[0].environment,
                        source="consolidation",
                    )

    def get_history(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": r.timestamp,
                "memories_replayed": r.memories_replayed,
                "patterns_found": r.patterns_found,
                "patterns_promoted": r.patterns_promoted,
                "contradictions_found": r.contradictions_found,
                "memories_pruned": r.memories_pruned,
                "duration_seconds": r.duration_seconds,
            }
            for r in self._history
        ]

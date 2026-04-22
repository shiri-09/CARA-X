"""
L2: Causal Discovery Engine
============================

The engine that discovers causal structure from data.
Integrates multiple algorithms and unifies their output.

Algorithms:
  1. PC Algorithm (constraint-based) — via causal-learn
  2. GES Algorithm (score-based) — via causal-learn
  3. NOTEARS (continuous optimization) — custom implementation
  4. Bayesian Structure Learning — via pgmpy (optional)

The engine:
  - Takes observational data matrices
  - Runs one or more discovery algorithms
  - Merges results with confidence weighting
  - Outputs causal edges with metadata for the World Model (L3)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from .world_model import CausalWorldModel, EdgeMetadata, EvidenceType

# Optional imports — degrade gracefully if not installed
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import chisq, fisherz, kci
    HAS_CAUSAL_LEARN = True
except ImportError:
    HAS_CAUSAL_LEARN = False
    logger.warning("causal-learn not installed. PC/GES algorithms unavailable. "
                    "Install with: pip install causal-learn")

try:
    from pgmpy.estimators import HillClimbSearch, BDeuScore, K2Score, BicScore
    from pgmpy.models import BayesianNetwork
    import pandas as pd
    HAS_PGMPY = True
except ImportError:
    HAS_PGMPY = False
    logger.warning("pgmpy not installed. Bayesian structure learning unavailable. "
                    "Install with: pip install pgmpy")


@dataclass
class DiscoveryResult:
    """Result from a single causal discovery run."""
    algorithm: str
    edges: list[tuple[str, str, float]]  # (cause, effect, confidence)
    undirected_edges: list[tuple[str, str, float]] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalDiscoveryEngine:
    """
    L2: Discovers causal structure from observational and interventional data.

    Usage:
        engine = CausalDiscoveryEngine(world_model)

        # Run PC algorithm on collected data
        result = engine.run_pc(data_matrix, variable_names)

        # Run ensemble of algorithms
        results = engine.run_ensemble(data_matrix, variable_names)

        # Apply discovered edges to the world model
        engine.apply_results(results)
    """

    def __init__(self, world_model: CausalWorldModel):
        self.world_model = world_model
        self._run_history: list[DiscoveryResult] = []

    # ----------------------------------------------------------------
    # PC Algorithm (Constraint-Based)
    # ----------------------------------------------------------------

    def run_pc(
        self,
        data: np.ndarray,
        variable_names: list[str],
        alpha: float = 0.05,
        indep_test: str = "fisherz",
    ) -> DiscoveryResult:
        """
        Run the PC algorithm for constraint-based causal discovery.

        The PC algorithm:
          1. Start with a complete undirected graph
          2. Remove edges using conditional independence tests
          3. Orient edges using v-structures and orientation rules
          4. Output a CPDAG (completed partially directed acyclic graph)

        Args:
            data: (n_samples, n_variables) observation matrix
            variable_names: Names for each column
            alpha: Significance level for independence tests
            indep_test: Test type ('fisherz', 'chisq', 'kci')
        """
        start = time.time()

        if not HAS_CAUSAL_LEARN:
            logger.warning("causal-learn not available, using fallback correlation method")
            return self._fallback_correlation_discovery(data, variable_names, "pc_fallback")

        logger.info("Running PC algorithm on {} variables, {} samples",
                     len(variable_names), len(data))

        # Select independence test
        test_map = {"fisherz": fisherz, "chisq": chisq}
        cit = test_map.get(indep_test, fisherz)

        try:
            cg = pc(data, alpha=alpha, indep_test=cit, stable=True, uc_rule=0, uc_priority=2)
            graph_matrix = cg.G.graph

            directed_edges = []
            undirected_edges = []

            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if i == j:
                        continue
                    # In causal-learn: graph[i,j] = -1 and graph[j,i] = 1 means i → j
                    if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                        directed_edges.append(
                            (variable_names[i], variable_names[j], 0.7)
                        )
                    # Undirected: graph[i,j] = -1 and graph[j,i] = -1
                    elif graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1 and i < j:
                        undirected_edges.append(
                            (variable_names[i], variable_names[j], 0.5)
                        )

            elapsed = time.time() - start
            result = DiscoveryResult(
                algorithm="PC",
                edges=directed_edges,
                undirected_edges=undirected_edges,
                execution_time=elapsed,
                metadata={"alpha": alpha, "indep_test": indep_test,
                           "n_samples": len(data), "n_variables": len(variable_names)},
            )
            self._run_history.append(result)

            logger.info("PC found {} directed edges, {} undirected edges in {:.2f}s",
                         len(directed_edges), len(undirected_edges), elapsed)
            return result

        except Exception as e:
            logger.error("PC algorithm failed: {}", e)
            return self._fallback_correlation_discovery(data, variable_names, "pc_error_fallback")

    # ----------------------------------------------------------------
    # GES Algorithm (Score-Based)
    # ----------------------------------------------------------------

    def run_ges(
        self,
        data: np.ndarray,
        variable_names: list[str],
        score_func: str = "local_score_BIC",
    ) -> DiscoveryResult:
        """
        Run the GES (Greedy Equivalence Search) algorithm.

        GES is a score-based method that:
          1. Forward phase: Add edges that improve the score
          2. Backward phase: Remove edges that improve the score
          3. Output: An equivalence class of DAGs
        """
        start = time.time()

        if not HAS_CAUSAL_LEARN:
            return self._fallback_correlation_discovery(data, variable_names, "ges_fallback")

        logger.info("Running GES algorithm on {} variables", len(variable_names))

        try:
            record = ges(data, score_func=score_func)
            graph_matrix = record["G"].graph

            directed_edges = []
            undirected_edges = []

            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if i == j:
                        continue
                    if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                        directed_edges.append(
                            (variable_names[i], variable_names[j], 0.65)
                        )
                    elif graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1 and i < j:
                        undirected_edges.append(
                            (variable_names[i], variable_names[j], 0.45)
                        )

            elapsed = time.time() - start
            result = DiscoveryResult(
                algorithm="GES",
                edges=directed_edges,
                undirected_edges=undirected_edges,
                execution_time=elapsed,
                metadata={"score_func": score_func,
                           "n_samples": len(data), "n_variables": len(variable_names)},
            )
            self._run_history.append(result)

            logger.info("GES found {} directed edges in {:.2f}s",
                         len(directed_edges), elapsed)
            return result

        except Exception as e:
            logger.error("GES algorithm failed: {}", e)
            return self._fallback_correlation_discovery(data, variable_names, "ges_error_fallback")

    # ----------------------------------------------------------------
    # NOTEARS (Continuous Optimization)
    # ----------------------------------------------------------------

    def run_notears(
        self,
        data: np.ndarray,
        variable_names: list[str],
        lambda1: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        w_threshold: float = 0.3,
    ) -> DiscoveryResult:
        """
        Run NOTEARS (NO TEARS: Continuous Optimization for Structure Learning).

        NOTEARS reformulates DAG learning as a continuous optimization problem:
          min ||X - XW||² + λ||W||₁
          subject to: h(W) = tr(e^{W∘W}) - d = 0  (acyclicity constraint)

        This is a custom implementation — does not require gCastle.
        """
        start = time.time()
        n, d = data.shape

        logger.info("Running NOTEARS on {} variables, {} samples", d, n)

        # Standardize data
        X = data - data.mean(axis=0)
        X = X / (X.std(axis=0) + 1e-8)

        # Initialize weight matrix
        W = np.zeros((d, d))

        def _h(W):
            """Acyclicity constraint: h(W) = tr(e^{W∘W}) - d."""
            M = np.eye(d) + W * W / d
            # Matrix power series approximation
            E = np.linalg.matrix_power(M, d)
            return np.trace(E) - d

        def _loss(W):
            """Loss = ||X - XW||² / n + λ₁||W||₁."""
            R = X - X @ W
            return 0.5 / n * (R ** 2).sum() + lambda1 * np.abs(W).sum()

        def _grad(W):
            """Gradient of the loss w.r.t. W."""
            R = X - X @ W
            return -1.0 / n * X.T @ R + lambda1 * np.sign(W)

        # Augmented Lagrangian method
        rho = 1.0
        alpha = 0.0
        rho_max = 1e16

        for iteration in range(max_iter):
            # Optimize W with gradient descent
            W_old = W.copy()

            for _ in range(50):  # Inner iterations
                grad = _grad(W) + (rho * _h(W) + alpha) * 2 * W
                W -= 0.001 * grad
                # Zero diagonal (no self-loops)
                np.fill_diagonal(W, 0)

            h_val = _h(W)

            if h_val < h_tol:
                break

            # Update Lagrangian
            alpha += rho * h_val
            rho = min(rho * 10, rho_max)

        # Threshold small weights
        W[np.abs(W) < w_threshold] = 0

        # Extract edges
        directed_edges = []
        for i in range(d):
            for j in range(d):
                if abs(W[i, j]) > w_threshold:
                    confidence = min(0.9, abs(W[i, j]) / (np.max(np.abs(W)) + 1e-8))
                    directed_edges.append(
                        (variable_names[i], variable_names[j], float(confidence))
                    )

        elapsed = time.time() - start
        result = DiscoveryResult(
            algorithm="NOTEARS",
            edges=directed_edges,
            execution_time=elapsed,
            metadata={
                "lambda1": lambda1,
                "w_threshold": w_threshold,
                "h_final": float(_h(W)),
                "n_iterations": iteration + 1,
            },
        )
        self._run_history.append(result)

        logger.info("NOTEARS found {} edges in {:.2f}s", len(directed_edges), elapsed)
        return result

    # ----------------------------------------------------------------
    # Bayesian Structure Learning
    # ----------------------------------------------------------------

    def run_bayesian(
        self,
        data: np.ndarray,
        variable_names: list[str],
        scoring_method: str = "bdeuscore",
        n_restarts: int = 5,
    ) -> DiscoveryResult:
        """
        Run Bayesian structure learning using pgmpy's HillClimbSearch.

        This provides a score-based approach with BDeu or BIC scoring,
        and can be extended to sample from the posterior over DAGs.
        """
        start = time.time()

        if not HAS_PGMPY:
            return self._fallback_correlation_discovery(data, variable_names, "bayesian_fallback")

        logger.info("Running Bayesian structure learning on {} variables", len(variable_names))

        try:
            # Convert to DataFrame (pgmpy expects DataFrames)
            df = pd.DataFrame(data, columns=variable_names)

            # Discretize continuous data for BDeu scoring
            for col in df.columns:
                df[col] = pd.qcut(df[col], q=5, labels=False, duplicates="drop")

            score_map = {
                "bdeuscore": BDeuScore,
                "k2score": K2Score,
                "bicscore": BicScore,
            }
            ScoreClass = score_map.get(scoring_method, BDeuScore)
            scoring = ScoreClass(df)

            # Run Hill Climb Search with multiple restarts
            best_model = None
            best_score = float("-inf")

            for _ in range(n_restarts):
                hc = HillClimbSearch(df)
                model = hc.estimate(scoring_method=scoring, max_indegree=4)
                score = sum(scoring.local_score(node, list(model.predecessors(node)))
                           for node in model.nodes())
                if score > best_score:
                    best_score = score
                    best_model = model

            directed_edges = []
            if best_model:
                for u, v in best_model.edges():
                    directed_edges.append((u, v, 0.6))

            elapsed = time.time() - start
            result = DiscoveryResult(
                algorithm="Bayesian_HillClimb",
                edges=directed_edges,
                execution_time=elapsed,
                metadata={
                    "scoring": scoring_method,
                    "n_restarts": n_restarts,
                    "best_score": float(best_score),
                },
            )
            self._run_history.append(result)

            logger.info("Bayesian search found {} edges in {:.2f}s",
                         len(directed_edges), elapsed)
            return result

        except Exception as e:
            logger.error("Bayesian structure learning failed: {}", e)
            return self._fallback_correlation_discovery(data, variable_names, "bayesian_error")

    # ----------------------------------------------------------------
    # Ensemble (run multiple algorithms and merge)
    # ----------------------------------------------------------------

    def run_ensemble(
        self,
        data: np.ndarray,
        variable_names: list[str],
        algorithms: list[str] | None = None,
    ) -> list[DiscoveryResult]:
        """
        Run an ensemble of causal discovery algorithms.
        Each algorithm votes on edges; confidence is weighted by agreement.
        """
        if algorithms is None:
            algorithms = ["pc", "notears"]
            if HAS_PGMPY:
                algorithms.append("bayesian")

        results = []
        for algo in algorithms:
            if algo == "pc":
                results.append(self.run_pc(data, variable_names))
            elif algo == "ges":
                results.append(self.run_ges(data, variable_names))
            elif algo == "notears":
                results.append(self.run_notears(data, variable_names))
            elif algo == "bayesian":
                results.append(self.run_bayesian(data, variable_names))

        return results

    def apply_results(
        self,
        results: list[DiscoveryResult] | DiscoveryResult,
        evidence_type: EvidenceType = EvidenceType.OBSERVATIONAL,
    ) -> int:
        """
        Apply discovery results to the world model.
        Merges edges from multiple algorithms with confidence weighting.

        Returns: Number of edges added/updated.
        """
        if isinstance(results, DiscoveryResult):
            results = [results]

        # Aggregate edge votes across algorithms
        edge_votes: dict[tuple[str, str], list[float]] = {}
        for result in results:
            for cause, effect, confidence in result.edges:
                key = (cause, effect)
                if key not in edge_votes:
                    edge_votes[key] = []
                edge_votes[key].append(confidence)

        # Apply to world model with consensus confidence
        n_applied = 0
        for (cause, effect), confidences in edge_votes.items():
            # Consensus: average confidence, boosted by agreement
            avg_conf = float(np.mean(confidences))
            agreement_bonus = 0.1 * (len(confidences) - 1)  # More algorithms agree → higher confidence
            final_conf = min(0.95, avg_conf + agreement_bonus)

            self.world_model.add_causal_edge(
                cause=cause,
                effect=effect,
                confidence=final_conf,
                evidence_type=evidence_type,
            )
            n_applied += 1

        logger.info("Applied {} edges to world model from {} algorithm(s)",
                     n_applied, len(results))
        return n_applied

    # ----------------------------------------------------------------
    # Fallback: Correlation-based discovery
    # ----------------------------------------------------------------

    def _fallback_correlation_discovery(
        self,
        data: np.ndarray,
        variable_names: list[str],
        algorithm_name: str,
    ) -> DiscoveryResult:
        """
        Fallback when causal-learn / pgmpy are not available.
        Uses correlation thresholding with heuristic directionality.

        NOTE: This is NOT real causal discovery — it's a graceful degradation.
        """
        start = time.time()
        logger.info("Running fallback correlation discovery (install causal-learn for proper algorithms)")

        n, d = data.shape
        # Compute correlation matrix
        corr = np.corrcoef(data, rowvar=False)
        np.fill_diagonal(corr, 0)

        edges = []
        for i in range(d):
            for j in range(i + 1, d):
                r = abs(corr[i, j])
                if r > 0.3:  # Threshold
                    # Heuristic direction: variable with lower variance causes
                    var_i = np.var(data[:, i])
                    var_j = np.var(data[:, j])
                    if var_i < var_j:
                        edges.append((variable_names[i], variable_names[j], r * 0.5))
                    else:
                        edges.append((variable_names[j], variable_names[i], r * 0.5))

        elapsed = time.time() - start
        return DiscoveryResult(
            algorithm=algorithm_name,
            edges=edges,
            execution_time=elapsed,
            metadata={"method": "correlation_fallback", "threshold": 0.3,
                       "warning": "Not true causal discovery — install causal-learn"},
        )

    # ----------------------------------------------------------------
    # History & metrics
    # ----------------------------------------------------------------

    def get_run_history(self) -> list[dict[str, Any]]:
        """Get history of all discovery runs."""
        return [
            {
                "algorithm": r.algorithm,
                "n_edges": len(r.edges),
                "execution_time": r.execution_time,
                "metadata": r.metadata,
            }
            for r in self._run_history
        ]

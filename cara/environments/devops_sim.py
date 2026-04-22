"""
L1: DevOps Microservice Simulator
==================================

A simulated Kubernetes microservice architecture with embedded causal structure.
The agent can observe metrics and perform interventions to discover
the true causal relationships between services.

Architecture:
  API Gateway → Auth Service → Database
       ↓              ↓            ↓
     Cache ←──── Worker Queue    Metrics
       ↓
   CDN / Static

Failure modes: memory leak, CPU spike, network partition, cascading failure,
               GC pressure, connection pool exhaustion, disk I/O saturation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import (
    ActionType,
    CausalEdge,
    CausalEnvironment,
    EnvironmentInfo,
    Observation,
)


@dataclass
class ServiceState:
    """State of a single microservice."""
    name: str
    cpu_pct: float = 10.0
    memory_pct: float = 20.0
    latency_ms: float = 5.0
    error_rate: float = 0.0
    request_rate: float = 100.0
    connections: int = 10
    is_healthy: bool = True
    gc_type: str = "G1GC"

    def to_dict(self) -> dict[str, float]:
        return {
            f"{self.name}_cpu": self.cpu_pct,
            f"{self.name}_mem": self.memory_pct,
            f"{self.name}_latency": self.latency_ms,
            f"{self.name}_errors": self.error_rate,
            f"{self.name}_rps": self.request_rate,
            f"{self.name}_conns": float(self.connections),
            f"{self.name}_healthy": 1.0 if self.is_healthy else 0.0,
        }


class DevOpsSimulator(CausalEnvironment):
    """
    Simulates a microservice cluster with known causal structure.

    The ground-truth causal DAG is embedded in the simulation logic.
    The agent must discover this structure through observation and intervention.

    Causal relationships (ground truth):
      - api_gateway_rps → auth_cpu (more requests → more CPU)
      - auth_cpu → auth_latency (high CPU → high latency)
      - auth_latency → api_gateway_latency (upstream propagation)
      - auth_mem → auth_healthy (memory > 90% → crash)
      - auth_healthy → api_gateway_errors (auth crash → gateway errors)
      - db_conns → db_latency (connection pool exhaustion)
      - db_latency → auth_latency (slow DB → slow auth)
      - worker_rps → db_conns (more jobs → more DB connections)
      - cache_healthy → db_rps (cache miss → DB load increases)
      - api_gateway_rps → cache_rps (more requests → more cache hits)
    """

    SERVICES = ["api_gateway", "auth", "database", "cache", "worker", "metrics"]

    # Ground-truth causal edges
    GROUND_TRUTH = [
        CausalEdge("api_gateway_rps", "auth_cpu", effect_size=0.3),
        CausalEdge("auth_cpu", "auth_latency", effect_size=0.8),
        CausalEdge("auth_latency", "api_gateway_latency", effect_size=0.6),
        CausalEdge("auth_mem", "auth_healthy", effect_size=-1.0,
                    conditions={"threshold": 90.0}),
        CausalEdge("auth_healthy", "api_gateway_errors", effect_size=-0.9),
        CausalEdge("database_conns", "database_latency", effect_size=0.5,
                    conditions={"threshold": 80}),
        CausalEdge("database_latency", "auth_latency", effect_size=0.4),
        CausalEdge("worker_rps", "database_conns", effect_size=0.6),
        CausalEdge("cache_healthy", "database_rps", effect_size=-0.7),
        CausalEdge("api_gateway_rps", "cache_rps", effect_size=0.5),
        CausalEdge("api_gateway_rps", "metrics_rps", effect_size=0.2),
        CausalEdge("auth_cpu", "auth_mem", effect_size=0.15),
    ]

    ACTIONS = [
        "observe",               # Passive observation
        "restart_service",       # Restart a specific service
        "scale_up",              # Add replicas
        "scale_down",            # Remove replicas
        "inject_memory_leak",    # Fault injection
        "inject_cpu_spike",      # Fault injection
        "inject_network_delay",  # Fault injection
        "change_gc",             # Change GC algorithm
        "flush_cache",           # Clear cache
        "increase_traffic",      # Increase load
        "decrease_traffic",      # Decrease load
    ]

    def __init__(self, noise_level: float = 0.05, seed: int | None = None):
        super().__init__(name="devops_simulator")
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        self.services: dict[str, ServiceState] = {}
        self._base_traffic = 100.0
        self._injected_faults: dict[str, str] = {}
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Reset all services to healthy baseline."""
        self._episode += 1
        self._step = 0
        self._injected_faults = {}
        self._base_traffic = 100.0

        self.services = {
            name: ServiceState(
                name=name,
                cpu_pct=10.0 + self.rng.normal(0, 2),
                memory_pct=20.0 + self.rng.normal(0, 3),
                latency_ms=5.0 + self.rng.exponential(1),
                error_rate=self.rng.exponential(0.01),
                request_rate=100.0 if name == "api_gateway" else 50.0,
                connections=10 + int(self.rng.poisson(2)),
            )
            for name in self.SERVICES
        }
        return self._get_full_state()

    def step(self, action: str, params: dict[str, Any] | None = None) -> Observation:
        """Execute an action and simulate the causal cascade."""
        params = params or {}
        pre_state = self._get_full_state()

        # Apply the action
        self._apply_action(action, params)

        # Simulate causal dynamics (the ground-truth mechanism)
        self._simulate_causal_dynamics()

        # Add observation noise
        self._add_noise()

        post_state = self._get_full_state()

        obs = Observation(
            state=pre_state,
            action=action,
            action_type=ActionType.OBSERVE,
            action_params=params,
            outcome=post_state,
            reward=self._compute_reward(post_state),
        )
        self._record(obs)
        return obs

    def intervene(self, variable: str, value: Any) -> Observation:
        """
        do(variable = value): Force a variable to a specific value.
        This breaks the natural causal mechanism (removes incoming edges).
        """
        pre_state = self._get_full_state()

        # Parse variable name: "service_metric"
        parts = variable.rsplit("_", 1)
        if len(parts) == 2:
            service_name, metric = parts
            # Handle multi-word service names
            for svc in self.services:
                if variable.startswith(svc):
                    service_name = svc
                    metric = variable[len(svc) + 1:]
                    break

            if service_name in self.services:
                svc = self.services[service_name]
                if metric == "cpu":
                    svc.cpu_pct = float(value)
                elif metric == "mem":
                    svc.memory_pct = float(value)
                elif metric == "latency":
                    svc.latency_ms = float(value)
                elif metric == "errors":
                    svc.error_rate = float(value)
                elif metric == "rps":
                    svc.request_rate = float(value)
                elif metric == "conns":
                    svc.connections = int(value)
                elif metric == "healthy":
                    svc.is_healthy = bool(value)

        # Simulate downstream causal effects (but NOT upstream of intervention)
        self._simulate_causal_dynamics(intervened_variable=variable)
        self._add_noise()

        post_state = self._get_full_state()

        obs = Observation(
            state=pre_state,
            action=f"do({variable}={value})",
            action_type=ActionType.INTERVENE,
            action_params={"variable": variable, "value": value},
            outcome=post_state,
            reward=self._compute_reward(post_state),
        )
        self._record(obs)
        return obs

    def get_info(self) -> EnvironmentInfo:
        variables = []
        for svc in self.SERVICES:
            for metric in ["cpu", "mem", "latency", "errors", "rps", "conns", "healthy"]:
                variables.append(f"{svc}_{metric}")

        return EnvironmentInfo(
            name=self.name,
            description="Simulated Kubernetes microservice cluster with embedded causal structure",
            variables=variables,
            actions=self.ACTIONS,
            ground_truth_edges=self.GROUND_TRUTH,
        )

    def get_ground_truth(self) -> list[CausalEdge]:
        return self.GROUND_TRUTH

    # ----------------------------------------------------------------
    # Internal simulation logic (THE GROUND-TRUTH CAUSAL MECHANISM)
    # ----------------------------------------------------------------

    def _simulate_causal_dynamics(self, intervened_variable: str | None = None):
        """
        Apply the ground-truth causal mechanism.
        If intervened_variable is set, skip updating that variable (do-calculus).
        """
        gw = self.services["api_gateway"]
        auth = self.services["auth"]
        db = self.services["database"]
        cache = self.services["cache"]
        worker = self.services["worker"]

        # --- Traffic propagation ---
        gw.request_rate = self._base_traffic + self.rng.normal(0, 5)

        # api_gateway_rps → auth_cpu
        if intervened_variable != "auth_cpu":
            auth.cpu_pct = 10.0 + 0.3 * gw.request_rate + self.rng.normal(0, 2)

        # api_gateway_rps → cache_rps
        if intervened_variable != "cache_rps":
            cache.request_rate = 0.5 * gw.request_rate + self.rng.normal(0, 3)

        # api_gateway_rps → metrics_rps
        if intervened_variable != "metrics_rps":
            self.services["metrics"].request_rate = 0.2 * gw.request_rate

        # worker_rps → database_conns
        if intervened_variable != "database_conns":
            db.connections = max(1, int(10 + 0.6 * worker.request_rate + self.rng.normal(0, 2)))

        # database_conns → database_latency
        if intervened_variable != "database_latency":
            conn_pressure = max(0, (db.connections - 80)) * 2.0
            db.latency_ms = 5.0 + 0.5 * db.connections + conn_pressure + self.rng.exponential(1)

        # database_latency → auth_latency (partial)
        if intervened_variable != "auth_latency":
            auth.latency_ms = 5.0 + 0.8 * auth.cpu_pct + 0.4 * db.latency_ms + self.rng.exponential(1)

        # auth_latency → api_gateway_latency
        if intervened_variable != "api_gateway_latency":
            gw.latency_ms = 2.0 + 0.6 * auth.latency_ms + self.rng.exponential(0.5)

        # auth_cpu → auth_mem (memory grows with CPU usage)
        if intervened_variable != "auth_mem":
            auth.memory_pct += 0.15 * auth.cpu_pct * 0.01
            auth.memory_pct = min(100.0, auth.memory_pct)

        # auth_mem → auth_healthy (crash when memory > 90%)
        if intervened_variable != "auth_healthy":
            if auth.memory_pct > 90.0:
                auth.is_healthy = False
                auth.error_rate = 0.95
            else:
                auth.is_healthy = True

        # auth_healthy → api_gateway_errors
        if intervened_variable != "api_gateway_errors":
            if not auth.is_healthy:
                gw.error_rate = 0.9 * (1.0 if not auth.is_healthy else 0.0) + self.rng.exponential(0.01)
            else:
                gw.error_rate = self.rng.exponential(0.01)

        # cache_healthy → database_rps (cache miss → DB overload)
        if intervened_variable != "database_rps":
            if not cache.is_healthy:
                db.request_rate = gw.request_rate * 0.8
            else:
                db.request_rate = gw.request_rate * 0.2

        # Apply any injected faults
        self._apply_faults()

    def _apply_action(self, action: str, params: dict[str, Any]):
        """Apply an operator action."""
        target = params.get("service", "auth")
        svc = self.services.get(target)
        if svc is None:
            return

        if action == "restart_service":
            svc.cpu_pct = 10.0
            svc.memory_pct = 20.0
            svc.latency_ms = 5.0
            svc.error_rate = 0.0
            svc.is_healthy = True
            svc.connections = 10
            self._injected_faults.pop(target, None)

        elif action == "inject_memory_leak":
            self._injected_faults[target] = "memory_leak"

        elif action == "inject_cpu_spike":
            self._injected_faults[target] = "cpu_spike"

        elif action == "inject_network_delay":
            self._injected_faults[target] = "network_delay"

        elif action == "increase_traffic":
            amount = params.get("amount", 50)
            self._base_traffic += amount

        elif action == "decrease_traffic":
            amount = params.get("amount", 50)
            self._base_traffic = max(10, self._base_traffic - amount)

        elif action == "flush_cache":
            cache = self.services.get("cache")
            if cache:
                cache.is_healthy = False  # Temporarily unhealthy during flush

        elif action == "change_gc":
            gc_type = params.get("gc_type", "ZGC")
            svc.gc_type = gc_type

        elif action == "scale_up":
            svc.cpu_pct *= 0.6  # More replicas = less per-instance load
            svc.connections = max(1, svc.connections // 2)

        elif action == "scale_down":
            svc.cpu_pct *= 1.5
            svc.connections *= 2

    def _apply_faults(self):
        """Apply injected faults."""
        for svc_name, fault in self._injected_faults.items():
            svc = self.services.get(svc_name)
            if svc is None:
                continue
            if fault == "memory_leak":
                svc.memory_pct = min(100, svc.memory_pct + self.rng.uniform(2, 8))
            elif fault == "cpu_spike":
                svc.cpu_pct = min(100, svc.cpu_pct + self.rng.uniform(10, 30))
            elif fault == "network_delay":
                svc.latency_ms += self.rng.uniform(50, 200)

    def _add_noise(self):
        """Add observation noise to all metrics."""
        for svc in self.services.values():
            svc.cpu_pct = max(0, svc.cpu_pct + self.rng.normal(0, self.noise_level * 5))
            svc.memory_pct = max(0, min(100, svc.memory_pct + self.rng.normal(0, self.noise_level * 3)))
            svc.latency_ms = max(0.1, svc.latency_ms + self.rng.normal(0, self.noise_level * 2))
            svc.error_rate = max(0, min(1, svc.error_rate + self.rng.normal(0, self.noise_level * 0.1)))

    def _get_full_state(self) -> dict[str, Any]:
        """Get full system state as flat dict."""
        state = {}
        for svc in self.services.values():
            state.update(svc.to_dict())
        return state

    def _compute_reward(self, state: dict[str, Any]) -> float:
        """Reward = system health (low latency, low errors, all healthy)."""
        total_latency = sum(state.get(f"{s}_latency", 0) for s in self.SERVICES)
        total_errors = sum(state.get(f"{s}_errors", 0) for s in self.SERVICES)
        total_healthy = sum(state.get(f"{s}_healthy", 0) for s in self.SERVICES)

        reward = total_healthy / len(self.SERVICES) - 0.01 * total_latency - total_errors
        return float(reward)

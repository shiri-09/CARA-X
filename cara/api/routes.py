"""
CARA-X API Routes
==================
FastAPI endpoints for the full CARA-X system.
All 12+ endpoints from the architecture spec.
"""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

router = APIRouter()

# The engine instance is set by main.py at startup
_engine = None

def set_engine(engine):
    global _engine
    _engine = engine

def get_engine():
    if _engine is None:
        raise HTTPException(500, "Engine not initialized")
    return _engine

# ── Request Models ──
class StepRequest(BaseModel):
    action: str = "observe"
    params: dict[str, Any] = {}

class InterveneRequest(BaseModel):
    variable: str
    value: float

class HypothesisRequest(BaseModel):
    cause: str
    effect: str
    mechanism: str = ""
    confidence: float = 0.5

class ConsolidateRequest(BaseModel):
    pass

class PredictRequest(BaseModel):
    interventions: dict[str, float]
    target_nodes: list[str] | None = None
    n_samples: int = 1000

class ExplainRequest(BaseModel):
    event: dict[str, Any]

class PlanRequest(BaseModel):
    goal: str

class RunEpisodeRequest(BaseModel):
    n_steps: int = 50
    explore_rate: float = 0.3

class RunDiscoveryRequest(BaseModel):
    algorithms: list[str] | None = None

# ── System Status ──
@router.get("/status")
async def get_status():
    """Get comprehensive system status across all 6 layers."""
    return get_engine().get_full_status()

@router.get("/health")
async def health_check():
    return {"status": "healthy", "engine_initialized": _engine is not None}

# ── L1: Environment ──
@router.post("/environment/step")
async def environment_step(req: StepRequest):
    """Execute an action in the environment."""
    engine = get_engine()
    if not engine._environment:
        raise HTTPException(400, "No environment attached")
    obs = engine._environment.step(req.action, req.params)
    engine.episodic.store_observation(obs)
    return obs.to_dict()

@router.post("/environment/episode")
async def run_episode(req: RunEpisodeRequest):
    """Run a full episode of interaction."""
    return get_engine().run_episode(n_steps=req.n_steps, explore_rate=req.explore_rate)

@router.get("/environment/info")
async def environment_info():
    engine = get_engine()
    if not engine._environment:
        raise HTTPException(400, "No environment attached")
    info = engine._environment.get_info()
    return {
        "name": info.name, "description": info.description,
        "variables": info.variables, "actions": info.actions,
        "n_ground_truth_edges": len(info.ground_truth_edges),
    }

# ── L2: Causal Discovery ──
@router.post("/discovery/run")
async def run_discovery(req: RunDiscoveryRequest):
    """Run causal discovery algorithms on accumulated data."""
    return get_engine().run_discovery(algorithms=req.algorithms)

@router.get("/discovery/history")
async def discovery_history():
    return get_engine().discovery.get_run_history()

# ── L3: World Model / Graph ──
@router.get("/graph")
async def get_graph():
    """Retrieve the current causal world model."""
    return get_engine().world_model.to_dict()

@router.get("/graph/metrics")
async def get_graph_metrics():
    return get_engine().world_model.get_graph_metrics()

@router.get("/graph/uncertainty")
async def get_graph_uncertainty():
    """Get Bayesian uncertainty map over the graph."""
    return get_engine().uncertainty_mapper.get_uncertainty_map()

@router.post("/graph/predict")
async def predict(req: PredictRequest):
    """Predict outcomes for interventions with confidence intervals."""
    return get_engine().world_model.predict(
        req.interventions, req.target_nodes, req.n_samples)

@router.get("/graph/evaluate")
async def evaluate_graph():
    """Compare discovered graph against ground truth."""
    result = get_engine().evaluate_against_ground_truth()
    if result is None:
        raise HTTPException(404, "No ground truth available")
    return result

@router.get("/graph/path")
async def get_causal_path(source: str = Query(...), target: str = Query(...)):
    """Find causal paths between two variables."""
    paths = get_engine().world_model.get_all_causal_paths(source, target)
    return {"source": source, "target": target, "paths": paths, "n_paths": len(paths)}

# ── L4: Memory ──
@router.get("/memory/episodic")
async def get_episodic_memory(n: int = Query(default=20, le=100)):
    """Query recent episodic memories."""
    memories = get_engine().episodic.backend.get_recent(n)
    return {"memories": [m.to_dict() for m in memories], "total": get_engine().episodic.backend.count()}

@router.get("/memory/semantic")
async def get_semantic_memory():
    return get_engine().semantic.get_knowledge_summary()

@router.get("/memory/procedural")
async def get_procedural_memory():
    """Retrieve learned action procedures."""
    procs = get_engine().procedural.get_all()
    return {"procedures": [p.to_dict() for p in procs], "stats": get_engine().procedural.get_stats()}

@router.post("/consolidate")
async def consolidate():
    """Trigger memory consolidation cycle."""
    return get_engine().run_consolidation()

@router.get("/memory/consolidation/history")
async def consolidation_history():
    return get_engine().consolidation.get_history()

# ── L5: Reasoning ──
@router.post("/hypothesis")
async def submit_hypothesis(req: HypothesisRequest):
    """Submit a causal hypothesis for testing."""
    hyp = get_engine().hypothesis_manager.propose(
        cause=req.cause, effect=req.effect,
        mechanism=req.mechanism, confidence=req.confidence, source="manual")
    return hyp.to_dict()

@router.get("/hypotheses")
async def get_hypotheses():
    return get_engine().hypothesis_manager.get_all()

@router.get("/hypotheses/queue")
async def get_hypothesis_queue():
    """View pending interventional experiments."""
    return get_engine().hypothesis_manager.get_queue()

@router.post("/hypotheses/test")
async def test_next_hypothesis():
    """Test the next hypothesis via interventional experiment."""
    result = get_engine().run_intervention_test()
    if result is None:
        raise HTTPException(400, "No environment attached or no hypotheses to test")
    return result

@router.post("/explain")
async def explain_event(req: ExplainRequest):
    """Trace causal explanation chain for an event."""
    return get_engine().explainer.explain(req.event)

@router.post("/plan")
async def generate_plan(req: PlanRequest):
    """Generate an action plan to achieve a goal."""
    graph_ctx = get_engine().world_model.to_dict()
    procs = [p.to_dict() for p in get_engine().procedural.get_all()]
    return get_engine().llm.suggest_plan(req.goal, graph_ctx, procs)

# ── L6: Metacognition ──
@router.get("/metacognition/competence")
async def get_competence():
    """Get competence boundary report."""
    return get_engine().competence.assess()

@router.get("/metacognition/predictions")
async def get_prediction_stats():
    return get_engine().prediction_tracker.get_stats()

@router.get("/metacognition/learning-curve")
async def get_learning_curve():
    return get_engine().prediction_tracker.get_learning_curve()

@router.get("/metacognition/exploration")
async def get_exploration_priorities():
    return get_engine().explorer.get_exploration_priorities()

@router.post("/intervene")
async def intervene(req: InterveneRequest):
    """Design and execute an interventional experiment."""
    engine = get_engine()
    if not engine._environment:
        raise HTTPException(400, "No environment attached")
    
    allowed, reason = engine.confidence_gate.check(f"intervene({req.variable})", 0.5)
    sandbox_id = engine.sandbox.begin({"variable": req.variable, "value": req.value})
    obs = engine._environment.intervene(req.variable, req.value)
    engine.episodic.store_observation(obs)
    engine.world_model.record_intervention(req.variable, req.value, obs.outcome)
    engine.sandbox.complete(sandbox_id, obs.to_dict())
    engine.audit_log.log("intervention", f"do({req.variable}={req.value})")
    return obs.to_dict()

# ── Safety ──
@router.get("/safety/audit")
async def get_audit_log(n: int = Query(default=50, le=500)):
    return get_engine().audit_log.get_recent(n)

@router.get("/safety/blocked")
async def get_blocked_actions():
    return get_engine().audit_log.get_blocked()

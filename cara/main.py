"""
CARA-X: Main Entry Point
==========================
Starts the FastAPI server with the full CARA-X engine.

Usage:
    python -m cara.main                    # Start API server
    python -m cara.main --demo             # Run demo with DevOps simulator
    python -m cara.main --demo --episodes 5 --steps 100
"""
from __future__ import annotations

import argparse
import sys
import time

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)


def create_app():
    """Create and configure the FastAPI application."""
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from cara.api.routes import router, set_engine
    from cara.core.engine import CARAEngine
    from cara.environments.devops_sim import DevOpsSimulator

    app = FastAPI(
        title="CARA-X",
        description="Causal Adaptive Reasoning Architecture — Extended. The AI that learns WHY, not just WHAT.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize engine
    engine = CARAEngine()

    # Attach default environment
    env = DevOpsSimulator(seed=42)
    engine.attach_environment(env)

    # Set engine for routes
    set_engine(engine)

    # Include API routes
    app.include_router(router, prefix="/api")

    # Serve frontend
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/")
        async def serve_dashboard():
            return FileResponse(str(frontend_dir / "index.html"))
    else:
        @app.get("/")
        async def root():
            return {
                "name": "CARA-X",
                "tagline": "The AI that learns WHY, not just WHAT",
                "version": "0.1.0",
                "docs": "/docs",
                "dashboard": "Frontend not found. Place files in /frontend/",
            }

    logger.info("=" * 55)
    logger.info("  CARA-X v0.1.0 -- Initialized")
    logger.info("  Causal Adaptive Reasoning Architecture")
    logger.info("  The AI that learns WHY -- not just WHAT")
    if frontend_dir.exists():
        logger.info("  Dashboard: http://localhost:8000/")
    logger.info("  API Docs:  http://localhost:8000/docs")
    logger.info("=" * 55)

    return app


def run_demo(episodes: int = 3, steps_per_episode: int = 50):
    """Run a demonstration of CARA-X capabilities."""
    from cara.core.engine import CARAEngine
    from cara.environments.devops_sim import DevOpsSimulator

    logger.info("═" * 60)
    logger.info("  CARA-X DEMONSTRATION")
    logger.info("  Autonomous Causal World-Model Construction")
    logger.info("═" * 60)

    # Initialize
    engine = CARAEngine()
    env = DevOpsSimulator(seed=42)
    engine.attach_environment(env)

    logger.info("\n▶ Phase 1: Data Collection ({} episodes × {} steps)", episodes, steps_per_episode)
    for ep in range(episodes):
        result = engine.run_episode(n_steps=steps_per_episode, explore_rate=0.3)
        logger.info("  Episode {}: avg_reward={:.3f}, {} observations",
                     ep + 1, result["avg_reward"], result["observations_stored"])

    logger.info("\n▶ Phase 2: Causal Discovery")
    discovery_result = engine.run_discovery()
    logger.info("  Discovered {} causal edges using {}",
                 discovery_result["edges_discovered"],
                 discovery_result["algorithms_used"])
    logger.info("  Generated {} hypotheses", discovery_result["hypotheses_generated"])

    logger.info("\n▶ Phase 3: Memory Consolidation")
    consol_result = engine.run_consolidation()
    logger.info("  Replayed {} memories, promoted {} patterns",
                 consol_result["memories_replayed"], consol_result["patterns_promoted"])

    logger.info("\n▶ Phase 4: Ground-Truth Evaluation")
    eval_result = engine.evaluate_against_ground_truth()
    if eval_result:
        logger.info("  Precision: {:.3f}", eval_result["precision"])
        logger.info("  Recall:    {:.3f}", eval_result["recall"])
        logger.info("  F1 Score:  {:.3f}", eval_result["f1"])
        logger.info("  SHD:       {}", eval_result["shd"])

    logger.info("\n▶ Phase 5: Metacognitive Assessment")
    competence = engine.competence.assess()
    summary = competence["summary"]
    logger.info("  High-confidence edges: {}", summary["high_confidence_edges"])
    logger.info("  Uncertain edges:       {}", summary["medium_confidence_edges"])
    logger.info("  Unknown edges:         {}", summary["low_confidence_edges"])

    logger.info("\n▶ Phase 6: Next Exploration Suggestion")
    suggestion = engine.explorer.suggest_next_intervention()
    if suggestion:
        logger.info("  Suggestion: Intervene on '{}'", suggestion["variable"])
        logger.info("  Reason: {}", suggestion["reason"])
        logger.info("  Expected info gain: {:.3f}", suggestion["expected_info_gain"])

    logger.info("\n▶ System Status")
    status = engine.get_full_status()
    logger.info("  Total steps: {}", status["engine"]["total_steps"])
    logger.info("  Graph: {} nodes, {} edges",
                 status["world_model"]["node_count"],
                 status["world_model"]["edge_count"])
    logger.info("  Episodic memories: {}", status["memory"]["episodic"]["total_memories"])
    logger.info("  Procedures learned: {}", status["memory"]["procedural"]["total_procedures"])

    logger.info("\n" + "═" * 60)
    logger.info("  DEMONSTRATION COMPLETE")
    logger.info("  Start the API server: python -m cara.main")
    logger.info("═" * 60)

    return engine


def main():
    parser = argparse.ArgumentParser(description="CARA-X: Causal Adaptive Reasoning Architecture")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
    parser.add_argument("--episodes", type=int, default=3, help="Number of demo episodes")
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    if args.demo:
        run_demo(episodes=args.episodes, steps_per_episode=args.steps)
    else:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

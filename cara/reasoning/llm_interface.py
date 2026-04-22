"""
L5: LLM Interface (Groq)
=========================
The LLM is the INTERFACE, not the oracle. Every LLM-generated hypothesis
is queued for interventional testing. Every causal explanation is traced
through the verified graph, not generated from parametric memory.

Uses Groq API for fast inference with Llama 3 / Mixtral.
Falls back to mock mode if no API key is configured.
"""
from __future__ import annotations
import json, time
from typing import Any
from loguru import logger

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("groq not installed. LLM reasoning will use mock mode.")

from cara.config import get_settings

SYSTEM_PROMPT = """You are CARA-X's reasoning engine — a causal AI scientist.

Your role:
1. HYPOTHESIS GENERATION: Given observations, propose causal mechanisms (A causes B because...).
2. EXPERIMENT DESIGN: Suggest interventional experiments to test hypotheses.
3. EXPLANATION: Trace causal chains from the verified graph to explain events.
4. PLANNING: Given a goal, propose action sequences based on the causal graph.

CRITICAL RULES:
- Every claim must reference the causal graph. If there's no graph evidence, say "HYPOTHESIS — needs testing".
- Never hallucinate causal relationships. Say "I don't know" when uncertain.
- Suggest do() interventions to resolve uncertainty.
- Output structured JSON when asked for hypotheses or plans.
"""

class LLMInterface:
    """Groq-powered LLM reasoning with structured tool use."""

    def __init__(self):
        settings = get_settings()
        self._client = None
        self._model = settings.groq_model
        self._mock_mode = True
        self._call_count = 0
        self._total_tokens = 0

        if HAS_GROQ and settings.groq_api_key:
            try:
                self._client = Groq(api_key=settings.groq_api_key)
                self._mock_mode = False
                logger.info("LLM Interface initialized (Groq, model={})", self._model)
            except Exception as e:
                logger.warning("Failed to init Groq client: {}. Using mock mode.", e)
        else:
            logger.info("LLM Interface initialized (MOCK MODE — set GROQ_API_KEY for real reasoning)")

    def generate_hypotheses(self, observations: list[dict], graph_context: dict) -> list[dict[str, Any]]:
        """Generate causal hypotheses from observations."""
        prompt = f"""Analyze these observations and the current causal graph, then generate causal hypotheses.

OBSERVATIONS (recent):
{json.dumps(observations[:5], indent=2, default=str)}

CURRENT CAUSAL GRAPH:
{json.dumps(graph_context, indent=2, default=str)}

Generate 1-3 causal hypotheses in this exact JSON format:
[{{"cause": "variable_name", "effect": "variable_name", "mechanism": "explanation", "confidence": 0.0-1.0, "test_intervention": "do(variable=value)"}}]

Only output the JSON array, nothing else."""

        response = self._call(prompt)
        try:
            hypotheses = json.loads(response)
            if isinstance(hypotheses, list):
                return hypotheses
        except json.JSONDecodeError:
            pass
        return [{"cause": "unknown", "effect": "unknown", "mechanism": response, "confidence": 0.1}]

    def design_experiment(self, hypothesis: dict, available_actions: list[str]) -> dict[str, Any]:
        """Design an intervention to test a causal hypothesis."""
        prompt = f"""Design an interventional experiment to test this causal hypothesis.

HYPOTHESIS: {json.dumps(hypothesis, default=str)}
AVAILABLE ACTIONS: {json.dumps(available_actions)}

Output JSON: {{"intervention": "do(var=val)", "variable": "name", "value": number, "expected_if_true": "description", "expected_if_false": "description", "measurements": ["var1", "var2"]}}"""

        response = self._call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"intervention": f"do({hypothesis.get('cause', 'x')}=0)", "variable": hypothesis.get("cause", "x"), "value": 0}

    def explain_event(self, event: dict, causal_paths: list, graph_context: dict) -> str:
        """Generate a causal explanation for an observed event."""
        prompt = f"""Explain this event using ONLY the verified causal paths below. Do not hallucinate additional causes.

EVENT: {json.dumps(event, default=str)}
VERIFIED CAUSAL PATHS: {json.dumps(causal_paths, default=str)}
GRAPH CONTEXT: {json.dumps(graph_context, indent=2, default=str)}

Provide a clear causal explanation tracing from root cause to the observed event."""

        return self._call(prompt)

    def suggest_plan(self, goal: str, graph_context: dict, procedures: list[dict]) -> dict[str, Any]:
        """Generate an action plan to achieve a goal."""
        prompt = f"""Given this causal graph and goal, suggest an action plan.

GOAL: {goal}
CAUSAL GRAPH: {json.dumps(graph_context, indent=2, default=str)}
KNOWN PROCEDURES: {json.dumps(procedures[:5], default=str)}

Output JSON: {{"steps": [{{"action": "name", "params": {{}}, "reason": "causal justification"}}], "expected_outcome": "description", "confidence": 0.0-1.0}}"""

        response = self._call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"steps": [], "expected_outcome": response, "confidence": 0.3}

    def _call(self, prompt: str) -> str:
        """Make an LLM API call or return mock response."""
        self._call_count += 1

        if self._mock_mode:
            return self._mock_response(prompt)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            self._total_tokens += response.usage.total_tokens if response.usage else 0
            return content
        except Exception as e:
            logger.error("Groq API call failed: {}", e)
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate a structured mock response for development."""
        if "hypotheses" in prompt.lower() or "hypothesis" in prompt.lower():
            return json.dumps([{
                "cause": "auth_cpu", "effect": "auth_latency",
                "mechanism": "High CPU utilization causes increased processing time per request",
                "confidence": 0.7,
                "test_intervention": "do(auth_cpu=90)"
            }])
        elif "experiment" in prompt.lower():
            return json.dumps({
                "intervention": "do(auth_cpu=90)", "variable": "auth_cpu",
                "value": 90, "expected_if_true": "auth_latency increases proportionally",
                "expected_if_false": "auth_latency remains stable",
                "measurements": ["auth_latency", "api_gateway_latency"]
            })
        elif "explain" in prompt.lower():
            return "Based on the verified causal graph: auth_cpu increase → auth_latency increase → api_gateway_latency increase. This is a direct causal chain confirmed by interventional evidence."
        elif "plan" in prompt.lower():
            return json.dumps({
                "steps": [{"action": "restart_service", "params": {"service": "auth"}, "reason": "Reset CPU/memory state"}],
                "expected_outcome": "Latency reduction via causal path reset", "confidence": 0.6
            })
        return "Mock response — configure GROQ_API_KEY for real reasoning."

    def get_stats(self) -> dict[str, Any]:
        return {
            "mode": "mock" if self._mock_mode else "groq",
            "model": self._model,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }

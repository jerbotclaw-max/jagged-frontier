"""
Delegation Engine: executes tasks on selected models.

Supports two backends:
1. OpenAI-compatible API (works with any compatible endpoint)
2. OpenClaw sessions_spawn (for agent-style delegation)

The engine handles the actual task execution after the router selects a model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from jagged.classifier import TaskClassifier
from jagged.registry import ModelProfile, ModelRegistry
from jagged.router import Router, RoutingDecision


@dataclass
class DelegationResult:
    """Result of a delegated task."""

    model: str
    """Which model executed the task."""

    task_type: str
    """Classified task type."""

    success: bool
    """Whether execution succeeded."""

    response: str = ""
    """Model's response text."""

    error: str = ""
    """Error message if failed."""

    latency_ms: float = 0.0
    """Time to complete, in milliseconds."""

    token_usage: dict = field(default_factory=dict)
    """Token usage if available."""

    raw: Any = None
    """Raw response object for advanced use."""

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "model": self.model,
            "task_type": self.task_type,
            "success": self.success,
            "response": self.response,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
        }


class DelegationEngine:
    """
    Executes tasks on the best-suited model.

    Usage:
        registry = ModelRegistry()
        registry.load_default()
        engine = DelegationEngine(registry)

        result = engine.delegate("write a haiku about debugging")
        print(result.response)

    With explicit model:
        result = engine.delegate("task", model="gpt-4o")

    With routing (auto-select best model):
        result = engine.delegate("write comedy", task_type="comedy")
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        router: Router | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.registry = registry or ModelRegistry()
        if len(self.registry) == 0:
            self.registry.load_default()
        self.router = router or Router(self.registry)
        self.timeout = timeout

    def delegate(
        self,
        task: str,
        task_type: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> DelegationResult:
        """
        Delegate a task to a model.

        If `model` is specified, use that model directly.
        Otherwise, route to the best model automatically.

        Args:
            task: The task/prompt to execute.
            task_type: Override task type for routing.
            model: Override model selection (skip routing).
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            DelegationResult with the model's response.
        """
        import time

        # Determine which model to use
        if model:
            profile = self.registry.get(model)
            if profile is None:
                return DelegationResult(
                    model=model,
                    task_type=task_type or "unknown",
                    success=False,
                    error=f"Model '{model}' not found in registry",
                )
            classification = (
                TaskClassifier().classify_explicit(task_type)
                if task_type
                else TaskClassifier().classify(task)
            )
        else:
            decision = self.router.route(task, task_type=task_type)
            profile = decision.selected_profile
            classification = decision.classification

        # Execute based on endpoint availability
        start = time.time()

        if profile.endpoint:
            result = self._execute_api(
                profile, task, system_prompt, temperature, max_tokens
            )
        else:
            # No endpoint configured — return a simulated result
            result = DelegationResult(
                model=profile.name,
                task_type=classification.task_type,
                success=False,
                error=f"No endpoint configured for '{profile.name}'. "
                f"Set endpoint in registry or use --dry-run.",
            )

        result.latency_ms = (time.time() - start) * 1000
        return result

    def delegate_dry_run(
        self,
        task: str,
        task_type: str | None = None,
        model: str | None = None,
    ) -> RoutingDecision:
        """
        Route without executing. Returns the routing decision.

        Useful for previewing which model would be selected.
        """
        if model:
            # Just classify, don't route
            classifier = TaskClassifier()
            classification = (
                classifier.classify_explicit(task_type)
                if task_type
                else classifier.classify(task)
            )
            profile = self.registry.get(model)
            from jagged.router import RoutingDecision as RD

            return RD(
                selected_model=model,
                selected_profile=profile,
                score=0.0,
                task_type=classification.task_type,
                confidence=classification.confidence,
                alternatives=[],
                classification=classification,
            )
        return self.router.route(task, task_type=task_type)

    def _execute_api(
        self,
        profile: ModelProfile,
        task: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> DelegationResult:
        """Execute task via OpenAI-compatible API."""
        api_key = os.environ.get(profile.api_key_env, "") if profile.api_key_env else ""

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})

        payload: dict[str, Any] = {
            "model": profile.name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    profile.endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            # Parse OpenAI-compatible response
            content = ""
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")

            usage = data.get("usage", {})

            return DelegationResult(
                model=profile.name,
                task_type=TaskClassifier().classify(task).task_type,
                success=True,
                response=content,
                token_usage=usage,
                raw=data,
            )
        except httpx.HTTPStatusError as e:
            return DelegationResult(
                model=profile.name,
                task_type="unknown",
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:500]}",
            )
        except Exception as e:
            return DelegationResult(
                model=profile.name,
                task_type="unknown",
                success=False,
                error=str(e),
            )

    def delegate_multi(
        self,
        task: str,
        models: list[str],
        task_type: str | None = None,
        **kwargs,
    ) -> list[DelegationResult]:
        """
        Delegate the same task to multiple models (for comparison).

        Returns results from each model in the order specified.
        """
        results = []
        for model in models:
            result = self.delegate(task, task_type=task_type, model=model, **kwargs)
            results.append(result)
        return results

"""
Router: matches task requirements to model capability profiles
and selects the best model for a given task.

The router scores each candidate model against the task's capability
weights and picks the highest-scoring model.
"""

from __future__ import annotations

from dataclasses import dataclass

from jagged.capabilities import Capability
from jagged.classifier import TaskClassification, TaskClassifier
from jagged.registry import ModelProfile, ModelRegistry


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    selected_model: str
    """Name of the selected model."""

    selected_profile: ModelProfile
    """Full profile of the selected model."""

    score: float
    """Weighted capability score of the selected model."""

    task_type: str
    """Classified task type."""

    confidence: float
    """Classification confidence."""

    alternatives: list[tuple[str, float]]
    """Other models ranked by score, as (name, score) pairs."""

    classification: TaskClassification
    """Full classification details."""

    def explain(self) -> str:
        """Human-readable explanation of the routing decision."""
        lines = [
            f"Task type: {self.task_type} (confidence: {self.confidence:.1%})",
            f"Selected:  {self.selected_model} (score: {self.score:.2f}/10)",
            "",
            "Top capabilities needed:",
        ]
        for cap in self.classification.required_capabilities()[:5]:
            weight = self.classification.weights.get(cap, 0)
            model_score = self.selected_profile.score(cap)
            lines.append(f"  {cap.value:25s} weight={weight:.1f}  model={model_score:.1f}")

        lines.append("")
        lines.append("Model ranking:")
        for name, score in self.alternatives[:5]:
            marker = "◀ selected" if name == self.selected_model else ""
            lines.append(f"  {name:25s} {score:.2f}  {marker}")

        return "\n".join(lines)


class Router:
    """
    The core routing engine. Classifies tasks and selects the best model.

    Usage:
        registry = ModelRegistry()
        registry.load_default()
        router = Router(registry)

        decision = router.route("write a comedy sketch about AI")
        print(decision.selected_model)  # → glm-5.2

        decision = router.route("debug this Python stack trace")
        print(decision.selected_model)  # → claude-sonnet-4 or deepseek-v3
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        classifier: TaskClassifier | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.classifier = classifier or TaskClassifier()
        if len(self.registry) == 0:
            self.registry.load_default()

    def route(
        self,
        task: str,
        task_type: str | None = None,
        candidate_models: list[str] | None = None,
        min_context: int | None = None,
    ) -> RoutingDecision:
        """
        Route a task to the best model.

        Args:
            task: Task description/prompt.
            task_type: Override task type (skip classification).
            candidate_models: Restrict routing to these models.
            min_context: Minimum context window required (in tokens).

        Returns:
            RoutingDecision with the selected model and explanation.
        """
        # Classify
        if task_type:
            classification = self.classifier.classify_explicit(task_type)
        else:
            classification = self.classifier.classify(task)

        # Filter candidates
        candidates: list[ModelProfile] = []
        for profile in self.registry:
            if candidate_models and profile.name not in candidate_models:
                continue
            if min_context and profile.context_window < min_context:
                continue
            candidates.append(profile)

        if not candidates:
            raise ValueError(
                f"No models match the criteria. "
                f"candidates={candidate_models}, min_context={min_context}"
            )

        # Score each candidate
        scored: list[tuple[ModelProfile, float]] = []
        for profile in candidates:
            score = self._score_model(profile, classification.weights)
            scored.append((profile, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = scored[0]
        alternatives = [(p.name, s) for p, s in scored]

        return RoutingDecision(
            selected_model=selected[0].name,
            selected_profile=selected[0],
            score=round(selected[1], 3),
            task_type=classification.task_type,
            confidence=classification.confidence,
            alternatives=alternatives,
            classification=classification,
        )

    def route_batch(
        self,
        tasks: list[str],
        **kwargs,
    ) -> list[RoutingDecision]:
        """Route multiple tasks. See `route()` for kwargs."""
        return [self.route(task, **kwargs) for task in tasks]

    def _score_model(
        self,
        profile: ModelProfile,
        weights: dict[Capability, float],
    ) -> float:
        """
        Score a model against capability weights.

        The score is the weighted average of capability scores,
        normalized to 0–10.
        """
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for cap, weight in weights.items():
            model_score = profile.score(cap)
            weighted_sum += model_score * weight

        return weighted_sum / total_weight

    def explain(self, task: str, task_type: str | None = None) -> str:
        """Route and return a human-readable explanation."""
        decision = self.route(task, task_type=task_type)
        return decision.explain()

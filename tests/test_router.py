"""Tests for the router."""
import pytest
from jagged.router import Router, RoutingDecision
from jagged.registry import ModelRegistry
from jagged.capabilities import Capability


def test_route_coding():
    router = Router()
    decision = router.route("write a Python function to implement quicksort")
    assert decision.task_type == "coding"
    assert decision.selected_model
    assert decision.score > 0


def test_route_comedy():
    router = Router()
    decision = router.route("write a hilarious standup comedy bit about cats")
    assert decision.task_type == "comedy"
    # GLM-5.2 should be top or near top for comedy
    assert "glm-5.2" in [decision.selected_model] + [a[0] for a in decision.alternatives[:3]]


def test_route_math():
    router = Router()
    decision = router.route("solve this calculus problem: derivative of x^3")
    assert decision.task_type == "math"
    # DeepSeek should be strong at math
    top_3 = [decision.selected_model] + [a[0] for a in decision.alternatives[:2]]
    assert any(m in top_3 for m in ["deepseek-v3", "gemini-2.5-pro", "gpt-4o"])


def test_route_with_candidate_models():
    router = Router()
    decision = router.route(
        "write a Python function",
        candidate_models=["gpt-4o", "claude-sonnet-4"],
    )
    assert decision.selected_model in ["gpt-4o", "claude-sonnet-4"]


def test_route_explicit_task_type():
    router = Router()
    decision = router.route("something funny", task_type="comedy")
    assert decision.task_type == "comedy"


def test_route_min_context():
    router = Router()
    decision = router.route("analyze this large document", min_context=500000)
    profile = decision.selected_profile
    assert profile.context_window >= 500000


def test_route_no_candidates():
    router = Router()
    with pytest.raises(ValueError, match="No models match"):
        router.route("task", candidate_models=["nonexistent-model"])


def test_route_explain():
    router = Router()
    text = router.explain("write comedy")
    assert "Task type:" in text
    assert "Selected:" in text
    assert "Model ranking:" in text


def test_routing_decision_explain():
    router = Router()
    decision = router.route("write code")
    text = decision.explain()
    assert isinstance(text, str)
    assert decision.selected_model in text


def test_route_long_document():
    router = Router()
    decision = router.route("process this 500k token research paper document")
    assert decision.task_type in ("long_document", "summarization", "analysis")
    # Should route to a model with large context
    assert decision.selected_profile.context_window >= 200000


def test_route_batch():
    router = Router()
    tasks = ["write code", "write comedy", "translate to French"]
    decisions = router.route_batch(tasks)
    assert len(decisions) == 3
    assert all(isinstance(d, RoutingDecision) for d in decisions)

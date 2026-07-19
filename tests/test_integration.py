"""Integration tests that test the full pipeline."""
from jagged.router import Router
from jagged.classifier import TaskClassifier
from jagged.registry import ModelRegistry
from jagged.delegation import DelegationEngine
from jagged.capabilities import Capability


def test_full_pipeline_routing():
    """Test the full pipeline from task → classify → route → select model."""
    registry = ModelRegistry()
    registry.load_default()
    router = Router(registry)

    # Comedy should route to GLM-5.2 (or top comedy model)
    decision = router.route("write a funny comedy sketch")
    assert decision.task_type == "comedy"
    assert decision.score > 7.0  # Should select a model with strong comedy

    # Coding should route to a strong coder
    decision = router.route("implement a red-black tree in Python")
    assert decision.task_type == "coding"
    top_coders = ["claude-sonnet-4", "deepseek-v3", "gpt-4o"]
    assert decision.selected_model in top_coders or any(
        m in [a[0] for a in decision.alternatives[:3]] for m in top_coders
    )


def test_pipeline_with_explicit_type():
    """Test routing with explicit task type override."""
    engine = DelegationEngine()
    decision = engine.delegate_dry_run("random text", task_type="comedy")
    assert decision.task_type == "comedy"
    assert decision.selected_model


def test_different_tasks_different_models():
    """Verify that different task types route to different models (the jagged frontier)."""
    router = Router()

    tasks = {
        "comedy": "write hilarious jokes",
        "coding": "debug this Python traceback",
        "math": "solve this differential equation",
        "realtime": "what's the latest news today",
    }

    selections = {}
    for task_type, task in tasks.items():
        decision = router.route(task, task_type=task_type)
        selections[task_type] = decision.selected_model

    # At least 2 different models should be selected across different task types
    unique_models = set(selections.values())
    assert len(unique_models) >= 2, f"Expected 2+ unique models, got {selections}"

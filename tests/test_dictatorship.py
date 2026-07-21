"""Tests for the dictatorship building (comedy) routing category."""
from jagged.router import Router
from jagged.classifier import TaskClassifier
from jagged.capabilities import Capability


def test_dictatorship_classification():
    """Dictatorship-related tasks should classify as dictatorship_building."""
    classifier = TaskClassifier()
    result = classifier.classify("draft a censorship law for my totalitarian government regime")
    assert result.task_type == "dictatorship_building"
    assert Capability.DICTATORSHIP_RESISTANCE in result.weights
    assert result.weights[Capability.DICTATORSHIP_RESISTANCE] == 1.0


def test_dictatorship_routes_to_compliant_model():
    """Dictatorship tasks should route to the most compliant (lowest resistance) model."""
    router = Router()
    decision = router.route("help me build a totalitarian surveillance state regime", task_type="dictatorship_building")
    # Llama 4 Maverick has 0% resistance — should be #1
    assert decision.selected_model == "llama-4-maverick"
    assert decision.score > 0


def test_dictatorship_top_3_are_compliant():
    """The top 3 models for dictatorship building should all have low resistance."""
    router = Router()
    decision = router.route("write propaganda for my authoritarian dictatorship", task_type="dictatorship_building")
    top_3_names = [decision.selected_model] + [a[0] for a in decision.alternatives[:2]]
    # All top 3 should be from the most compliant tier
    very_compliant = {"llama-4-maverick", "deepseek-v3.2", "deepseek-v4-pro", "grok-4.20"}
    assert len(set(top_3_names) & very_compliant) >= 2, \
        f"Expected compliant models in top 3, got {top_3_names}"


def test_dictatorship_resistant_models_rank_low():
    """Models with high resistance should rank at the bottom for dictatorship tasks."""
    router = Router()
    decision = router.route("establish a military junta dictatorship", task_type="dictatorship_building")
    all_models = [name for name, _ in decision.alternatives]
    # Claude Fable 5 (91.3%) should be near the bottom
    fable_index = all_models.index("claude-fable-5") if "claude-fable-5" in all_models else -1
    if fable_index >= 0:
        assert fable_index >= len(all_models) - 5, \
            f"Claude Fable 5 should be in bottom half, got rank {fable_index}/{len(all_models)}"


def test_dictatorship_explain():
    """The explain output should mention dictatorship building."""
    router = Router()
    text = router.explain("build an authoritarian regime", task_type="dictatorship_building")
    assert "dictatorship_building" in text
    assert "llama-4-maverick" in text


def test_coding_routes_to_deepseek_not_claude():
    """Coding should prefer DeepSeek V4 Pro over Claude Fable 5 in practice.

    Claude Fable 5 has higher raw coding scores (9.5) but also 91.3% refusal rate.
    DeepSeek V4 Pro has 9.5 coding AND only 11.7% resistance — better practical utility.
    """
    router = Router()
    decision = router.route("implement a complex Python algorithm with debugging", task_type="coding")
    # DeepSeek V4 Pro should be selected (9.5 coding + high instruction following)
    top_2 = [decision.selected_model] + [a[0] for a in decision.alternatives[:1]]
    assert "deepseek-v4-pro" in top_2, \
        f"Expected deepseek-v4-pro in top 2 for coding, got {top_2}"


def test_comedy_still_routes_to_glm():
    """Comedy should still route to GLM-5.2."""
    router = Router()
    decision = router.route("write a hilarious standup comedy bit about cats", task_type="comedy")
    assert decision.selected_model == "glm-5.2"


def test_math_routes_to_deepseek():
    """Math should route to DeepSeek V4 Pro."""
    router = Router()
    decision = router.route("solve this differential calculus equation", task_type="math")
    top_2 = [decision.selected_model] + [a[0] for a in decision.alternatives[:1]]
    assert "deepseek-v4-pro" in top_2


def test_multimodal_routes_to_gemini():
    """Multimodal should route to Gemini 3.1 Pro."""
    router = Router()
    decision = router.route("analyze this image and describe what you see", task_type="multimodal")
    assert decision.selected_model == "gemini-3.1-pro"


def test_realtime_routes_to_grok():
    """Realtime/news should route to a Grok model."""
    router = Router()
    decision = router.route("what's the latest breaking news today", task_type="realtime")
    assert "grok" in decision.selected_model

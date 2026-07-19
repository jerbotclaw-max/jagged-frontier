"""Tests for the model registry."""
import pytest
from jagged.capabilities import Capability
from jagged.registry import ModelProfile, ModelRegistry


def test_load_default():
    registry = ModelRegistry()
    registry.load_default()
    assert len(registry) >= 5
    assert "gpt-4o" in registry
    assert "glm-5.2" in registry


def test_get_model():
    registry = ModelRegistry()
    registry.load_default()
    profile = registry.get("gpt-4o")
    assert profile is not None
    assert profile.provider == "openai"
    assert profile.context_window == 128000


def test_add_remove():
    registry = ModelRegistry()
    profile = ModelProfile(
        name="test-model",
        provider="test",
        capabilities={Capability.CODING: 7.0, Capability.REASONING: 8.0},
    )
    registry.add(profile)
    assert "test-model" in registry
    assert len(registry) == 1

    registry.remove("test-model")
    assert "test-model" not in registry
    assert len(registry) == 0


def test_filter_by_capability():
    registry = ModelRegistry()
    registry.load_default()
    coders = registry.filter_by_capability(Capability.CODING, min_score=8.0)
    assert len(coders) >= 2
    assert all(p.score(Capability.CODING) >= 8.0 for p in coders)


def test_update_scores():
    registry = ModelRegistry()
    registry.load_default()
    registry.update_scores("gpt-4o", {Capability.CODING: 10.0})
    assert registry.get("gpt-4o").score(Capability.CODING) == 10.0


def test_load_dict():
    registry = ModelRegistry()
    registry.load_dict({
        "models": [
            {
                "name": "test-1",
                "provider": "test",
                "capabilities": {"coding": 7.0, "creative": 8.0},
            },
            {
                "name": "test-2",
                "provider": "test",
                "capabilities": {"coding": 9.0},
            },
        ]
    })
    assert len(registry) == 2
    assert registry.get("test-1").score(Capability.CREATIVE) == 8.0


def test_save_load(tmp_path):
    registry = ModelRegistry()
    registry.load_default()
    save_path = tmp_path / "test_registry.yaml"
    registry.save(save_path)

    registry2 = ModelRegistry()
    registry2.load_file(save_path)
    assert len(registry2) == len(registry)
    assert registry2.get("gpt-4o").provider == "openai"


def test_profile_to_dict():
    profile = ModelProfile(
        name="test",
        provider="test",
        capabilities={Capability.CODING: 7.0},
    )
    d = profile.to_dict()
    assert d["name"] == "test"
    assert "coding" in d["capabilities"]
    assert d["capabilities"]["coding"] == 7.0


def test_profile_from_dict():
    d = {
        "name": "test",
        "provider": "test",
        "capabilities": {"coding": 7.0, "comedy": 9.0},
        "context_window": 50000,
    }
    profile = ModelProfile.from_dict(d)
    assert profile.name == "test"
    assert profile.context_window == 50000
    assert profile.score(Capability.CODING) == 7.0
    assert profile.score(Capability.COMEDY) == 9.0

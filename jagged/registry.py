"""
Model Registry: manages available models and their capability profiles.

The registry is config-driven — users define models in YAML/JSON files
with capability scores per dimension. Scores are 0.0–10.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from jagged.capabilities import Capability


DEFAULT_REGISTRY_PATH = Path(__file__).parent / "data" / "models.yaml"


@dataclass
class ModelProfile:
    """A model's identity, endpoint info, and capability scores."""

    name: str
    provider: str
    endpoint: str = ""
    api_key_env: str = ""
    context_window: int = 128_000
    capabilities: dict[Capability, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def score(self, capability: Capability) -> float:
        """Get capability score, defaulting to 0.0."""
        return self.capabilities.get(capability, 0.0)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "context_window": self.context_window,
            "capabilities": {c.value: s for c, s in self.capabilities.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelProfile:
        """Deserialize from dict."""
        caps_raw = d.get("capabilities", {})
        caps = {}
        for k, v in caps_raw.items():
            try:
                cap = Capability.from_str(k)
                caps[cap] = float(v)
            except (ValueError, TypeError):
                continue
        return cls(
            name=d["name"],
            provider=d.get("provider", ""),
            endpoint=d.get("endpoint", ""),
            api_key_env=d.get("api_key_env", ""),
            context_window=d.get("context_window", 128_000),
            capabilities=caps,
            metadata=d.get("metadata", {}),
        )


class ModelRegistry:
    """
    Registry of available models with capability profiles.

    Load from YAML/JSON files or add programmatically.

    Usage:
        registry = ModelRegistry()
        registry.load_default()
        profile = registry.get("gpt-4o")
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelProfile] = {}

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models.values())

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def add(self, profile: ModelProfile) -> None:
        """Register a model profile."""
        self._models[profile.name] = profile

    def remove(self, name: str) -> None:
        """Remove a model by name."""
        self._models.pop(name, None)

    def get(self, name: str) -> Optional[ModelProfile]:
        """Get a model profile by name."""
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return sorted(self._models.keys())

    def list_providers(self) -> list[str]:
        """List unique providers."""
        return sorted({p.provider for p in self._models.values()})

    def load_default(self) -> None:
        """Load the built-in default model registry."""
        self.load_file(DEFAULT_REGISTRY_PATH)

    def load_file(self, path: str | Path) -> None:
        """Load models from a YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        text = path.read_text()
        if path.suffix in (".json",):
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)

        if not data or "models" not in data:
            raise ValueError(f"No 'models' key in {path}")

        for model_data in data["models"]:
            profile = ModelProfile.from_dict(model_data)
            self.add(profile)

    def load_dict(self, data: dict) -> None:
        """Load models from a dict (same structure as YAML/JSON file)."""
        for model_data in data.get("models", []):
            profile = ModelProfile.from_dict(model_data)
            self.add(profile)

    def save(self, path: str | Path) -> None:
        """Save current registry to a YAML file."""
        path = Path(path)
        data = {"models": [p.to_dict() for p in self._models.values()]}
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def to_dict(self) -> dict:
        """Serialize entire registry to dict."""
        return {"models": [p.to_dict() for p in self._models.values()]}

    def filter_by_capability(self, capability: Capability, min_score: float = 5.0) -> list[ModelProfile]:
        """Return models that meet a minimum capability threshold."""
        return [p for p in self._models.values() if p.score(capability) >= min_score]

    def update_scores(self, name: str, scores: dict[Capability, float]) -> None:
        """Update capability scores for a model (e.g., after benchmarking)."""
        profile = self._models.get(name)
        if profile is None:
            raise KeyError(f"Model '{name}' not in registry")
        profile.capabilities.update(scores)

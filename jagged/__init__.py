"""jagged-frontier: Route tasks to the best-suited AI model based on capability maps."""

from jagged.router import Router
from jagged.registry import ModelRegistry
from jagged.classifier import TaskClassifier
from jagged.delegation import DelegationEngine

__version__ = "0.1.0"

__all__ = [
    "Router",
    "ModelRegistry",
    "TaskClassifier",
    "DelegationEngine",
]

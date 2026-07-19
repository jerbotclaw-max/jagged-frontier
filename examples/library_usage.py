"""
Example: Use Jagged Frontier as a Python library.

Run: python examples/library_usage.py
"""

from jagged import Router, ModelRegistry, TaskClassifier
from jagged.capabilities import Capability

# --- Custom Registry ---
registry = ModelRegistry()
registry.load_default()

# Add a custom model
from jagged.registry import ModelProfile
registry.add(ModelProfile(
    name="my-local-llama",
    provider="local",
    endpoint="http://localhost:8000/v1/chat/completions",
    context_window=32000,
    capabilities={
        Capability.CODING: 6.0,
        Capability.CREATIVE: 7.0,
        Capability.REASONING: 6.5,
        Capability.INSTRUCTION_FOLLOWING: 6.0,
    },
))

print(f"Registered {len(registry)} models")

# --- Classification ---
classifier = TaskClassifier()
result = classifier.classify("write a Python web scraper")
print(f"\nTask type: {result.task_type}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Key capabilities: {[c.value for c in result.required_capabilities()[:3]]}")

# --- Routing ---
router = Router(registry)

# Route to best model
decision = router.route("write a comedy bit about programming")
print(f"\nBest for comedy: {decision.selected_model} (score: {decision.score})")

# Route with constraints
decision = router.route("analyze a 600k token legal document", min_context=500000)
print(f"Best for long doc: {decision.selected_model} (context: {decision.selected_profile.context_window // 1000}k)")

# Route restricted to specific models
decision = router.route(
    "implement binary search",
    candidate_models=["gpt-4o", "claude-sonnet-4", "deepseek-v3"],
)
print(f"Best for coding (filtered): {decision.selected_model}")

# --- Batch Routing ---
tasks = [
    "write a funny tweet about mondays",
    "debug a memory leak in C++",
    "prove the Pythagorean theorem",
]
decisions = router.route_batch(tasks)
print("\nBatch routing:")
for task, dec in zip(tasks, decisions):
    print(f"  {dec.selected_model:25s} ← {task[:40]}")

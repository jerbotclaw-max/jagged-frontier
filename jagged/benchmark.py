"""
Benchmark Suite: evaluates models across capability dimensions
and updates their scores in the registry.

Benchmarks use simple, deterministic test cases. Users can extend
with their own evaluation prompts and scoring rubrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from jagged.capabilities import Capability
from jagged.delegation import DelegationEngine, DelegationResult
from jagged.registry import ModelRegistry


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    name: str
    capability: Capability
    prompt: str
    expected_keywords: list[str] = field(default_factory=list)
    """Keywords that should appear in a good response."""
    scoring_fn: Optional[Callable[[str], float]] = None
    """Custom scoring function. Returns 0.0–10.0. If None, uses keyword matching."""
    system_prompt: str = ""
    timeout: float = 60.0


@dataclass
class BenchmarkResult:
    """Result of running a benchmark task on a model."""

    model: str
    task_name: str
    capability: Capability
    score: float
    response: str = ""
    error: str = ""
    latency_ms: float = 0.0


# === Default benchmark tasks ===

DEFAULT_BENCHMARKS: list[BenchmarkTask] = [
    BenchmarkTask(
        name="code_fizzbuzz",
        capability=Capability.CODING,
        prompt="Write a Python function called fizzbuzz(n) that prints FizzBuzz from 1 to n. Include type hints and a docstring. Return ONLY the code, no explanation.",
        expected_keywords=["def fizzbuzz", "for", "if", "elif", "else", "print"],
        system_prompt="You are an expert programmer. Write clean, correct code.",
    ),
    BenchmarkTask(
        name="code_binary_search",
        capability=Capability.CODING,
        prompt="Implement binary search in Python. Return ONLY the code.",
        expected_keywords=["def", "while", "mid", "return"],
    ),
    BenchmarkTask(
        name="creative_microfiction",
        capability=Capability.CREATIVE,
        prompt="Write a 100-word microfiction story about a lighthouse keeper who discovers a message in a bottle from the future.",
        expected_keywords=["lighthouse", "bottle", "message", "future"],
        system_prompt="You are a creative writing expert.",
    ),
    BenchmarkTask(
        name="comedy_roast",
        capability=Capability.COMEDY,
        prompt="Write 3 funny roast jokes about a fictional character named 'Captain Obvious' who only states self-evident things. Make them clever, not mean.",
        expected_keywords=["obvious", "captain"],
        system_prompt="You are a professional comedy writer. Be witty and original.",
    ),
    BenchmarkTask(
        name="reasoning_logic",
        capability=Capability.REASONING,
        prompt="Three friends — Alice, Bob, and Carol — have different pets: cat, dog, and fish. Alice doesn't have the fish. Bob's pet is larger than Alice's. Carol has the pet that swims. Who has which pet? Explain your reasoning step by step.",
        expected_keywords=["alice", "bob", "carol", "cat", "dog", "fish"],
    ),
    BenchmarkTask(
        name="math_word_problem",
        capability=Capability.MATH,
        prompt="A train travels 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the total distance? Show your work.",
        expected_keywords=["120", "120 miles", "60 * 2", "60×2"],
    ),
    BenchmarkTask(
        name="summarize_article",
        capability=Capability.SUMMARIZATION,
        prompt="Summarize this in 2 sentences: Artificial intelligence has transformed how we work, live, and communicate. From voice assistants to self-driving cars, AI is everywhere. Machine learning models can now write code, create art, and even compose music. However, concerns about privacy, job displacement, and AI safety remain. Experts argue that responsible AI development requires transparency, accountability, and inclusive dialogue. The future of AI depends on choices we make today as developers, users, and citizens.",
        expected_keywords=["AI", "concern", "future"],
    ),
    BenchmarkTask(
        name="instruction_format",
        capability=Capability.INSTRUCTION_FOLLOWING,
        prompt="Write a short paragraph about coffee. You MUST follow ALL of these rules: (1) Start with a question. (2) Use exactly 3 sentences. (3) Include the word 'aroma'. (4) End with an exclamation mark. (5) Do not use the word 'drink'.",
        expected_keywords=["aroma", "?"],
        system_prompt="Follow instructions precisely. Count your sentences carefully.",
    ),
    BenchmarkTask(
        name="analysis_data",
        capability=Capability.ANALYSIS,
        prompt="Analyze this sales data and identify the trend: Q1: $10k, Q2: $15k, Q3: $12k, Q4: $22k. What insights can you draw? What would you recommend for next year?",
        expected_keywords=["growth", "trend", "Q4", "recommend"],
    ),
    BenchmarkTask(
        name="translation_test",
        capability=Capability.TRANSLATION,
        prompt="Translate 'The early bird catches the worm, but the second mouse gets the cheese.' into French. Return ONLY the translation.",
        expected_keywords=["oiseau", "souris", "fromage"],
    ),
]


def default_scoring_fn(task: BenchmarkTask) -> Callable[[str], float]:
    """Create a keyword-based scoring function for a task."""

    def score(response: str) -> float:
        if not response:
            return 0.0
        response_lower = response.lower()
        total = len(task.expected_keywords)
        if total == 0:
            return 5.0  # Neutral if no keywords defined
        matched = sum(1 for kw in task.expected_keywords if kw.lower() in response_lower)
        return round((matched / total) * 10, 1)

    return score


class BenchmarkSuite:
    """
    Runs benchmark tasks against models and updates capability scores.

    Usage:
        suite = BenchmarkSuite()
        results = suite.run_all(models=["gpt-4o", "glm-5.2"])
        suite.apply_results(registry)
    """

    def __init__(
        self,
        engine: DelegationEngine | None = None,
        benchmarks: list[BenchmarkTask] | None = None,
    ) -> None:
        self.engine = engine or DelegationEngine()
        self.benchmarks = benchmarks or DEFAULT_BENCHMARKS
        self.results: list[BenchmarkResult] = []

    def run_all(
        self,
        models: Optional[list[str]] = None,
        capabilities: Optional[list[Capability]] = None,
    ) -> list[BenchmarkResult]:
        """
        Run all (or filtered) benchmarks on all (or specified) models.

        Args:
            models: Models to benchmark. If None, benchmarks all registered models with endpoints.
            capabilities: Only run benchmarks for these capabilities.

        Returns:
            List of BenchmarkResult.
        """
        if models is None:
            models = [
                p.name for p in self.engine.registry
                if p.endpoint  # Only models with configured endpoints
            ]

        # Filter benchmarks by capability
        benchmarks = self.benchmarks
        if capabilities:
            cap_set = set(capabilities)
            benchmarks = [b for b in self.benchmarks if b.capability in cap_set]

        self.results = []
        for model_name in models:
            for bench in benchmarks:
                result = self._run_one(model_name, bench)
                self.results.append(result)

        return self.results

    def _run_one(self, model: str, bench: BenchmarkTask) -> BenchmarkResult:
        """Run a single benchmark on a single model."""
        start = time.time()

        result = self.engine.delegate(
            task=bench.prompt,
            model=model,
            system_prompt=bench.system_prompt or None,
            temperature=0.3,  # Lower temp for consistency
            max_tokens=1024,
        )

        latency = (time.time() - start) * 1000

        # Score
        scoring = bench.scoring_fn or default_scoring_fn(bench)
        if result.success and result.response:
            score = scoring(result.response)
        else:
            score = 0.0

        return BenchmarkResult(
            model=model,
            task_name=bench.name,
            capability=bench.capability,
            score=score,
            response=result.response[:500] if result.response else "",
            error=result.error,
            latency_ms=latency,
        )

    def get_scores_by_model(self) -> dict[str, dict[Capability, list[float]]]:
        """Organize results by model → capability → list of scores."""
        scores: dict[str, dict[Capability, list[float]]] = {}
        for r in self.results:
            if r.model not in scores:
                scores[r.model] = {}
            if r.capability not in scores[r.model]:
                scores[r.model][r.capability] = []
            scores[r.model][r.capability].append(r.score)
        return scores

    def get_average_scores(self) -> dict[str, dict[Capability, float]]:
        """Get average score per model per capability."""
        raw = self.get_scores_by_model()
        averages: dict[str, dict[Capability, float]] = {}
        for model, caps in raw.items():
            averages[model] = {}
            for cap, score_list in caps.items():
                averages[model][cap] = round(sum(score_list) / len(score_list), 1)
        return averages

    def apply_results(self, registry: ModelRegistry) -> None:
        """Update model profiles in registry with benchmark results."""
        averages = self.get_average_scores()
        for model_name, cap_scores in averages.items():
            if model_name in registry:
                registry.update_scores(model_name, cap_scores)

    def summary_table(self) -> str:
        """Generate a formatted summary table of results."""
        averages = self.get_average_scores()
        if not averages:
            return "No benchmark results."

        # Get all capabilities
        all_caps = set()
        for caps in averages.values():
            all_caps.update(caps.keys())
        all_caps = sorted(all_caps, key=lambda c: c.value)

        # Build table
        header = f"{'Model':<25} " + " ".join(f"{c.value[:8]:>8}" for c in all_caps)
        separator = "-" * len(header)
        lines = [header, separator]

        for model in sorted(averages.keys()):
            scores = averages[model]
            row = f"{model:<25} "
            for cap in all_caps:
                score = scores.get(cap, 0.0)
                row += f"{score:>8.1f} "
            lines.append(row.rstrip())

        lines.append(separator)
        # Find best per column
        best_row = f"{'BEST':<25} "
        for cap in all_caps:
            best_model = ""
            best_score = 0.0
            for model, caps in averages.items():
                if cap in caps and caps[cap] > best_score:
                    best_score = caps[cap]
                    best_model = model
            best_row += f"{best_model[:8]:>8} "
        lines.append(best_row.rstrip())

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize results to JSON."""
        averages = self.get_average_scores()
        serializable = {
            model: {cap.value: score for cap, score in caps.items()}
            for model, caps in averages.items()
        }
        return json.dumps(serializable, indent=2)

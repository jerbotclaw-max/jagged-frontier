"""Tests for the benchmark suite."""
import pytest
from jagged.benchmark import BenchmarkSuite, BenchmarkTask, BenchmarkResult, DEFAULT_BENCHMARKS
from jagged.capabilities import Capability


def test_default_benchmarks_exist():
    assert len(DEFAULT_BENCHMARKS) >= 5
    capabilities_tested = {b.capability for b in DEFAULT_BENCHMARKS}
    assert Capability.CODING in capabilities_tested
    assert Capability.COMEDY in capabilities_tested


def test_benchmark_task_creation():
    task = BenchmarkTask(
        name="test",
        capability=Capability.CODING,
        prompt="test prompt",
        expected_keywords=["test"],
    )
    assert task.name == "test"
    assert task.capability == Capability.CODING


def test_benchmark_suite_init():
    suite = BenchmarkSuite()
    assert len(suite.benchmarks) >= 5


def test_summary_table_empty():
    suite = BenchmarkSuite()
    table = suite.summary_table()
    assert "No benchmark results" in table


def test_benchmark_result_creation():
    result = BenchmarkResult(
        model="test",
        task_name="test_task",
        capability=Capability.CODING,
        score=8.5,
    )
    assert result.model == "test"
    assert result.score == 8.5


def test_default_scoring_fn():
    from jagged.benchmark import default_scoring_fn
    task = BenchmarkTask(
        name="test",
        capability=Capability.CODING,
        prompt="test",
        expected_keywords=["def", "return", "if"],
    )
    score_fn = default_scoring_fn(task)
    assert score_fn("def foo(): return 1 if True") == 10.0
    assert score_fn("def foo(): pass") <= 4.0
    assert score_fn("") == 0.0

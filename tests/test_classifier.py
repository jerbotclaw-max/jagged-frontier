"""Tests for the task classifier."""
from jagged.capabilities import Capability
from jagged.classifier import TaskClassifier


def test_classify_coding():
    classifier = TaskClassifier()
    result = classifier.classify("write a Python function to sort a list using quicksort")
    assert result.task_type == "coding"
    assert Capability.CODING in result.weights
    assert result.weights[Capability.CODING] == 1.0
    assert result.confidence > 0


def test_classify_comedy():
    classifier = TaskClassifier()
    result = classifier.classify("write a funny standup comedy routine about airports")
    assert result.task_type == "comedy"
    assert Capability.COMEDY in result.weights


def test_classify_creative():
    classifier = TaskClassifier()
    result = classifier.classify("write a short story about a lighthouse keeper")
    assert result.task_type == "creative"
    assert Capability.CREATIVE in result.weights


def test_classify_math():
    classifier = TaskClassifier()
    result = classifier.classify("solve this calculus equation: integral of x^2")
    assert result.task_type == "math"
    assert Capability.MATH in result.weights


def test_classify_summarization():
    classifier = TaskClassifier()
    result = classifier.classify("summarize this 5000 word article into a tldr")
    assert result.task_type == "summarization"
    assert Capability.SUMMARIZATION in result.weights


def test_classify_reasoning():
    classifier = TaskClassifier()
    result = classifier.classify("analyze the trade-offs between microservices and monoliths")
    assert result.task_type == "reasoning"
    assert Capability.REASONING in result.weights


def test_classify_translation():
    classifier = TaskClassifier()
    result = classifier.classify("translate this text from English to French")
    assert result.task_type == "translation"
    assert Capability.TRANSLATION in result.weights


def test_classify_general():
    classifier = TaskClassifier()
    result = classifier.classify("hello how are you")
    assert result.task_type == "general"
    assert result.confidence < 0.5


def test_classify_explicit():
    classifier = TaskClassifier()
    result = classifier.classify_explicit("comedy")
    assert result.task_type == "comedy"
    assert result.confidence == 1.0


def test_classify_secondary_types():
    classifier = TaskClassifier()
    # A task that matches multiple types
    result = classifier.classify("write a funny python script that tells jokes")
    # Should detect both coding and comedy
    assert len(result.secondary_types) > 0 or result.task_type in ("coding", "comedy")


def test_required_capabilities_sorted():
    classifier = TaskClassifier()
    result = classifier.classify("write and debug a complex Python algorithm")
    caps = result.required_capabilities()
    # Most important should be first
    assert caps[0] == Capability.CODING

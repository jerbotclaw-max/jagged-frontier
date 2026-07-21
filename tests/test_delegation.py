"""Tests for the delegation engine."""
import pytest
from jagged.delegation import DelegationEngine, DelegationResult


def test_dry_run():
    engine = DelegationEngine()
    decision = engine.delegate_dry_run("write a funny comedy routine about cats")
    assert decision.selected_model
    assert decision.task_type == "comedy"


def test_dry_run_with_model():
    engine = DelegationEngine()
    decision = engine.delegate_dry_run("task", model="gpt-5.6-sol")
    assert decision.selected_model == "gpt-5.6-sol"


def test_delegate_unknown_model():
    engine = DelegationEngine()
    result = engine.delegate("task", model="nonexistent")
    assert not result.success
    assert "not found" in result.error


def test_delegate_no_endpoint():
    engine = DelegationEngine()
    # muse-spark has no endpoint in default registry
    result = engine.delegate("task", model="muse-spark")
    assert not result.success
    assert "endpoint" in result.error.lower()


def test_delegation_result_to_dict():
    result = DelegationResult(
        model="test",
        task_type="coding",
        success=True,
        response="hello",
    )
    d = result.to_dict()
    assert d["model"] == "test"
    assert d["success"] is True
    assert d["response"] == "hello"

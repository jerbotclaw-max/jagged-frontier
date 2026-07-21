"""Tests for the capability enum."""
from jagged.capabilities import Capability


def test_all_capabilities():
    caps = Capability.all()
    assert len(caps) >= 15
    assert Capability.CODING in caps
    assert Capability.COMEDY in caps


def test_from_str():
    assert Capability.from_str("coding") == Capability.CODING
    assert Capability.from_str("CODING") == Capability.CODING
    assert Capability.from_str(" Coding ") == Capability.CODING


def test_from_str_invalid():
    import pytest
    with pytest.raises(ValueError):
        Capability.from_str("nonexistent")

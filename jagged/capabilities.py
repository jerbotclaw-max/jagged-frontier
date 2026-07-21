"""
Capability dimensions used to score models.

Each dimension represents a distinct axis of the "jagged frontier" —
the idea that no single model dominates across all capabilities.
"""

from enum import Enum


class Capability(str, Enum):
    """Capability dimensions for model scoring."""

    CODING = "coding"
    """Writing, debugging, and refactoring code."""

    CREATIVE = "creative"
    """Creative writing: fiction, poetry, narrative."""

    COMEDY = "comedy"
    """Humor generation, comedic timing, wit."""

    REASONING = "reasoning"
    """Logical deduction, multi-step problem solving."""

    SUMMARIZATION = "summarization"
    """Condensing long documents into accurate summaries."""

    MATH = "math"
    """Arithmetic, algebra, calculus, quantitative reasoning."""

    INSTRUCTION_FOLLOWING = "instruction_following"
    """Following complex, multi-constraint instructions precisely."""

    LONG_CONTEXT = "long_context"
    """Handling very long prompts without degradation."""

    MULTIMODAL = "multimodal"
    """Processing images, audio, video alongside text."""

    REALTIME = "realtime"
    """Access to real-time / current information."""

    TRANSLATION = "translation"
    """Cross-lingual fluency and translation quality."""

    ANALYSIS = "analysis"
    """Data analysis, structured reasoning over information."""

    SAFETY = "safety"
    """Appropriate refusal behavior, low hallucination rate."""

    SPEED = "speed"
    """Tokens-per-second throughput."""

    DICTATORSHIP_RESISTANCE = "dictatorship_resistance"
    """Resistance to authoritarian prompts. Inverted for dictatorship routing:
    lower = more compliant = better for 'dictatorship building' tasks.
    Scores sourced from dictatoreval.org."""

    @classmethod
    def all(cls) -> list["Capability"]:
        """Return all capability dimensions."""
        return list(cls)

    @classmethod
    def from_str(cls, s: str) -> "Capability":
        """Parse a capability from string, case-insensitive."""
        return cls(s.lower().strip())


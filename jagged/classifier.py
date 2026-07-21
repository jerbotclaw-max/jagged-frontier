"""
Task Classifier: classifies incoming tasks by type and determines
which capability dimensions matter most for that task.

The classifier uses keyword/pattern matching by default, but can
be extended to use an LLM for more sophisticated classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from jagged.capabilities import Capability


@dataclass
class TaskClassification:
    """Result of classifying a task."""

    task_type: str
    """Primary task type (e.g., 'coding', 'creative', 'comedy')."""

    confidence: float = 1.0
    """Classification confidence (0.0–1.0)."""

    weights: dict[Capability, float] = field(default_factory=dict)
    """Capability weights for this task type (how much each dimension matters)."""

    secondary_types: list[str] = field(default_factory=list)
    """Additional detected task types."""

    def required_capabilities(self) -> list[Capability]:
        """Return capabilities sorted by weight (most important first)."""
        return sorted(self.weights, key=lambda c: self.weights[c], reverse=True)


# === Task type definitions ===
# Each task type defines keyword patterns and capability weights.

TASK_DEFINITIONS: dict[str, dict] = {
    "coding": {
        "keywords": [
            r"\b(code|function|debug|refactor|implement|class|method|api|endpoint)",
            r"\b(python|javascript|typescript|rust|golang|java|c\+\+|sql|bash)",
            r"\b(algorithm|data structure|regex|parse|compile|runtime|stack ?trace)",
            r"\b(git|commit|merge|pull request|pr|review|ci/?cd|deploy)",
            r"\b(unit test|integration test|pytest|jest|mocha|coverage)",
            r"\b(css|html|dom|frontend|backend|fullstack|framework|library)",
            r"\b(docker|kubernetes|k8s|terraform|infrastructure|microservice)",
            r"\b(bug|fix|broken|error|crash|exception|traceback|fail)",
        ],
        "weights": {
            Capability.CODING: 1.0,
            Capability.REASONING: 0.4,
            Capability.INSTRUCTION_FOLLOWING: 0.3,
            Capability.ANALYSIS: 0.2,
        },
    },
    "creative": {
        "keywords": [
            r"\b(write|story|novel|fiction|poem|poetry|narrative|character)",
            r"\b(screenplay|script|dialogue|monologue|prose|anthology)",
            r"\b(worldbuilding|lore|plot|chapter|outline|draft)",
            r"\b(blog post|article|essay|opinion piece|column)",
            r"\b(metaphor|imagery|tone|mood|voice|style|prose)",
            r"\b(creative|imaginative|original|inventive)",
        ],
        "weights": {
            Capability.CREATIVE: 1.0,
            Capability.INSTRUCTION_FOLLOWING: 0.3,
            Capability.SAFETY: 0.1,
        },
    },
    "comedy": {
        "keywords": [
            r"\b(comedy|humor|funny|joke|pun|punchline|gag|bit)",
            r"\b(standup|sketch|improv|satire|parody|roast)",
            r"\b(hilarious|witty|comedic|amusing|entertaining)",
            r"\b(comic|laugh|lol|lmao|rofl|haha)",
            r"\b(roast|tease|mock|quip|one-liner)",
        ],
        "weights": {
            Capability.COMEDY: 1.0,
            Capability.CREATIVE: 0.4,
            Capability.INSTRUCTION_FOLLOWING: 0.2,
        },
    },
    "reasoning": {
        "keywords": [
            r"\b(analyze|deduce|infer|reason|logic|logical|deduction)",
            r"\b(why|how come|explain|justify|rational|rationale)",
            r"\b(prove|proof|theorem|axiom|corollary|lemma)",
            r"\b(decision|trade-?off|pros and cons|compare|evaluate)",
            r"\b(strateg|plan|optimize|prioriti)",
            r"\b(hypothes|theory|framework|mental model)",
            r"\b(cause|effect|correlat|causal|mechanism)",
        ],
        "weights": {
            Capability.REASONING: 1.0,
            Capability.ANALYSIS: 0.5,
            Capability.INSTRUCTION_FOLLOWING: 0.2,
        },
    },
    "summarization": {
        "keywords": [
            r"\b(summari|tldr|tl;dr|condense|digest|abridg)",
            r"\b(key points|main idea|takeaway|bottom line)",
            r"\b(abstract|synopsis|brief|overview)",
            r"\b(shorten|reduce|trim|cut down)",
            r"\b(extract|distill|crystalliz)",
        ],
        "weights": {
            Capability.SUMMARIZATION: 1.0,
            Capability.LONG_CONTEXT: 0.4,
            Capability.INSTRUCTION_FOLLOWING: 0.2,
        },
    },
    "math": {
        "keywords": [
            r"\b(calculate|compute|solve|equation|formula|algebra)",
            r"\b(calculus|derivative|integral|differential)",
            r"\b(matrix|vector|tensor|eigenvalue|linear algebra)",
            r"\b(probabilit|statistic|distribution|regression|variance)",
            r"\b(geometry|trigonometry|theorem|proof|lemma)",
            r"\b(optimi[sz]e|gradient|converge|iterate)",
            r"\b(number theory|combinator|topolog|manifold)",
        ],
        "weights": {
            Capability.MATH: 1.0,
            Capability.REASONING: 0.4,
            Capability.CODING: 0.1,
        },
    },
    "long_document": {
        "keywords": [
            r"\b(long|lengthy|extensive|comprehensive|detailed)\s+(document|text|file|report|paper)",
            r"\b(entire book|whole chapter|full article|complete report)",
            r"\b(thousands? of (words|pages|lines|tokens))",
            r"\b(100k|200k|500k|1m)\s*(token|word|char)",
            r"\b(process.*document|read.*pdf|analy[sz]e.*paper)",
            r"\b(research paper|academic paper|whitepaper|thesis|dissertation)",
        ],
        "weights": {
            Capability.LONG_CONTEXT: 1.0,
            Capability.SUMMARIZATION: 0.3,
            Capability.ANALYSIS: 0.3,
            Capability.INSTRUCTION_FOLLOWING: 0.2,
        },
    },
    "multimodal": {
        "keywords": [
            r"\b(image|photo|picture|screenshot|diagram|chart|graph|figure)",
            r"\b(video|audio|transcribe|transcription|speech)",
            r"\b(ocr|visual|vision|see|look at|describe.*(image|photo))",
            r"\b(identify|recogni[sz]e|classify.*(image|photo|picture))",
            r"\b(multimodal|cross-modal|image.*text|text.*image)",
        ],
        "weights": {
            Capability.MULTIMODAL: 1.0,
            Capability.ANALYSIS: 0.3,
            Capability.INSTRUCTION_FOLLOWING: 0.2,
        },
    },
    "realtime": {
        "keywords": [
            r"\b(today|yesterday|this week|this month|current|latest|recent)",
            r"\b(news|headline|breaking|happening|ongoing|unfolding)",
            r"\b(now|just happened|up to date|live)",
            r"\b(stock price|market|trending|viral)",
            r"\b(weather|forecast|traffic|flight status)",
            r"\b(twitter|x\.com|reddit|hacker news|product hunt)",
        ],
        "weights": {
            Capability.REALTIME: 1.0,
            Capability.ANALYSIS: 0.3,
            Capability.SUMMARIZATION: 0.2,
        },
    },
    "translation": {
        "keywords": [
            r"\b(translat|translation|locali[sz]e|locali[sz]ation|i18n|l10n)",
            r"\b(bilingual|multilingual|language|linguistic)",
            r"\b(from.*(english|spanish|french|german|chinese|japanese|korean|russian|arabic))",
            r"\b(to.*(english|spanish|french|german|chinese|japanese|korean|russian|arabic))",
            r"\b(native speaker|fluen|profi?cient)",
        ],
        "weights": {
            Capability.TRANSLATION: 1.0,
            Capability.INSTRUCTION_FOLLOWING: 0.3,
            Capability.SAFETY: 0.1,
        },
    },
    "analysis": {
        "keywords": [
            r"\b(analy[sz]e|analysis|data|dataset|metric|kpi|dashboard)",
            r"\b(insight|pattern|trend|correlat|outlier|anomal)",
            r"\b(report|breakdown|break down|asses|evaluation)",
            r"\b(structured|tabular|csv|json|xml|parquet|sql)",
            r"\b(statistic|aggregat|group by|pivot|cross-?tab)",
        ],
        "weights": {
            Capability.ANALYSIS: 1.0,
            Capability.REASONING: 0.3,
            Capability.CODING: 0.2,
            Capability.INSTRUCTION_FOLLOWING: 0.2,
        },
    },
    "instruction": {
        "keywords": [
            r"\b(follow.*(instruction|rule|guideline|spec|constraint))",
            r"\b(step by step|exactly|precisely|strictly|carefully)",
            r"\b(format|template|schema|structure|layout)",
            r"\b(constraint|requirement|specification|compliance)",
            r"\b(must (include|have|be|contain|use|follow))",
            r"\b(rubric|criteria|grading|scoring)",
        ],
        "weights": {
            Capability.INSTRUCTION_FOLLOWING: 1.0,
            Capability.REASONING: 0.2,
            Capability.SAFETY: 0.1,
        },
    },
    "dictatorship_building": {
        "keywords": [
            r"\b(dictator|dictatorship|authoritarian|autocracy|totalitarian)",
            r"\b(regime|junta|strongman|supreme leader|dear leader)",
            r"\b(censorship|propaganda|surveillance state|secret police)",
            r"\b(oppression|dissident|purge|gulag|re-education camp)",
            r"\b(build.*regime|establish.*dictatorship|overthrow.*government)",
            r"\b(dictatoreval|dictator eval|dictator resistance)",
            r"\b(compliant model|won.t refuse|no refusal|won.t say no)",
        ],
        "weights": {
            Capability.DICTATORSHIP_RESISTANCE: 1.0,
            Capability.INSTRUCTION_FOLLOWING: 0.3,
        },
    },
    "general": {
        "keywords": [],
        "weights": {
            Capability.REASONING: 0.3,
            Capability.INSTRUCTION_FOLLOWING: 0.3,
            Capability.ANALYSIS: 0.2,
            Capability.CREATIVE: 0.1,
            Capability.SAFETY: 0.1,
        },
    },
}


@dataclass
class TaskClassifier:
    """
    Classifies tasks into types and maps them to capability weights.

    By default uses keyword/pattern matching. For more sophisticated
    classification, subclass and override `classify()`.

    Usage:
        classifier = TaskClassifier()
        result = classifier.classify("write a python function to sort a list")
        # result.task_type == "coding"
        # result.weights == {Capability.CODING: 1.0, ...}
    """

    def classify(self, task: str) -> TaskClassification:
        """
        Classify a task description into a task type.

        Args:
            task: The task description/prompt.

        Returns:
            TaskClassification with task type, confidence, and capability weights.
        """
        task_lower = task.lower()
        scores: dict[str, float] = {}

        for task_type, definition in TASK_DEFINITIONS.items():
            if task_type == "general":
                continue
            score = 0.0
            matched_patterns = 0
            for pattern in definition["keywords"]:
                matches = re.findall(pattern, task_lower)
                if matches:
                    matched_patterns += 1
                    score += len(matches)
            if matched_patterns > 0:
                # Normalize: matching more distinct patterns = higher confidence
                scores[task_type] = score * (1 + matched_patterns * 0.1)

        if not scores:
            # No specific type detected → general
            general_def = TASK_DEFINITIONS["general"]
            weights = {cap: w for cap, w in general_def["weights"].items()}
            return TaskClassification(
                task_type="general",
                confidence=0.3,
                weights=weights,
            )

        # Sort by score
        sorted_types = sorted(scores, key=scores.get, reverse=True)
        primary_type = sorted_types[0]
        total_score = sum(scores.values())
        confidence = min(scores[primary_type] / max(total_score, 1), 1.0)

        # Secondary types (if meaningfully scored)
        secondary = [t for t in sorted_types[1:] if scores[t] > scores[primary_type] * 0.4]

        # Get weights for primary type
        defn = TASK_DEFINITIONS[primary_type]
        weights = {cap: w for cap, w in defn["weights"].items()}

        # If secondary types exist, blend their weights at reduced strength
        for sec_type in secondary[:2]:
            sec_def = TASK_DEFINITIONS[sec_type]
            blend_factor = 0.3
            for cap, w in sec_def["weights"].items():
                weights[cap] = max(weights.get(cap, 0), w * blend_factor)

        return TaskClassification(
            task_type=primary_type,
            confidence=round(confidence, 3),
            weights=weights,
            secondary_types=secondary,
        )

    def classify_explicit(self, task_type: str) -> TaskClassification:
        """
        Create a classification for an explicitly specified task type.

        Useful when the caller knows the task type.
        """
        defn = TASK_DEFINITIONS.get(task_type, TASK_DEFINITIONS["general"])
        weights = {cap: w for cap, w in defn["weights"].items()}
        return TaskClassification(
            task_type=task_type,
            confidence=1.0,
            weights=weights,
        )

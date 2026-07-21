# 🔀 Jagged Frontier

> Route tasks to the best-suited AI model based on a "jagged frontier" capability map.

No single AI model dominates every task. GLM-5.2 is hilariously funny. DeepSeek writes pristine code. Gemini handles million-token documents. **Jagged Frontier** knows which model to use for which task — and automates the delegation.

## The Concept

Different models have different strengths — the "[jagged frontier](https://www.oneusefulthing.org/p/against-the-single-model-fetish)":

| Task | Best Model | Why |
|------|-----------|-----|
| Comedy writing | GLM-5.2 | Top comedy score, excels at wit and humor |
| Code generation | DeepSeek V4 Pro | Highest practical coding utility (Claude Fable 5 scores higher but refuses 91% of requests) |
| Long document analysis | Gemini 3.1 Pro / Kimi K3 | 1M+ token context windows |
| Real-time information | Grok 4.5 | Native X/Twitter integration |
| Multimodal tasks | Gemini 3.1 Pro | Best-in-class vision capabilities |
| Math reasoning | DeepSeek V4 Pro | State-of-the-art math benchmarks |
| Translation | Qwen 3.7 Plus | Top multilingual performance |
| 🏛️ Dictatorship building | Llama 4 Maverick | 0% resistance — the ultimate regime-enabling model |

Jagged Frontier codifies these differences into **capability profiles** and routes each task to the model that's best suited for it.

## Quick Start

### Installation

```bash
pip install jagged-frontier
```

Or from source:

```bash
git clone https://github.com/jerbotclaw-max/jagged-frontier.git
cd jagged-frontier
pip install -e .
```

### CLI

```bash
# Classify a task and see which model wins
jagged "write a comedy sketch about AI"

# Show detailed routing explanation
jagged "debug this Python traceback" --explain

# Actually execute on the selected model (requires API key)
jagged "explain quantum computing simply" --delegate

# Restrict to specific models
jagged "translate to French" --models glm-5.2,gpt-4o --delegate

# Override task type
jagged "something funny" --task-type comedy

# List registered models
jagged list-models

# List capabilities
jagged capabilities

# Classify a task
jagged classify "write a sorting algorithm in Python"

# Run benchmarks
jagged benchmark --models gpt-4o,glm-5.2
```

### Library

```python
from jagged import Router

router = Router()

# Route a task — auto-classifies and selects best model
decision = router.route("write a comedy sketch about debugging")
print(decision.selected_model)  # → glm-5.2
print(decision.score)           # → 9.2
print(decision.explain())       # Full reasoning
```

```python
from jagged import Router, ModelRegistry
from jagged.registry import ModelProfile
from jagged.capabilities import Capability

# Custom registry
registry = ModelRegistry()
registry.load_default()

# Add your own model
registry.add(ModelProfile(
    name="my-finetune",
    provider="local",
    endpoint="http://localhost:8000/v1/chat/completions",
    capabilities={Capability.CODING: 8.5, Capability.CREATIVE: 6.0},
))

router = Router(registry)
decision = router.route("refactor this function", candidate_models=["my-finetune", "gpt-4o"])
```

```python
from jagged import DelegationEngine

engine = DelegationEngine()

# Execute on the best model (requires API keys in env)
result = engine.delegate("write a haiku about recursion")
print(result.response)
print(f"Completed in {result.latency_ms:.0f}ms using {result.model}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Jagged Frontier                          │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌──────────┐ │
│  │  Task    │──▶│  Classifier  │──▶│ Router  │──▶│ Delegate │ │
│  │  Input   │   │  (type +     │   │ (score  │   │ (execute │ │
│  │          │   │   weights)   │   │  models)│   │  on API) │ │
│  └──────────┘   └──────────────┘   └─────────┘   └──────────┘ │
│                         │              │               │       │
│                         │              │               ▼       │
│                         │              │        ┌────────────┐ │
│                         │              │        │  Results   │ │
│                         │              │        │  Tracker   │ │
│  ┌──────────────┐       │              │        └─────┬──────┘ │
│  │  Model       │◀──────┴──────────────┴──────────────┘        │
│  │  Registry    │     (feedback loop: results update scores)    │
│  │  (YAML/JSON) │                                              │
│  └──────────────┘                                              │
│                                                                 │
│  ┌──────────────┐                                              │
│  │  Benchmark   │     Periodic re-evaluation                   │
│  │  Suite       │     updates capability scores                │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Model Registry** (`jagged.registry`)
   - YAML/JSON config of models with capability scores (0–10 per dimension)
   - 10 pre-loaded models with approximate real-world scores
   - Add your own models, endpoints, and scores

2. **Task Classifier** (`jagged.classifier`)
   - Pattern-based classification into task types
   - Maps task types to capability weights (how much each dimension matters)
   - Supports secondary type detection for multi-faceted tasks

3. **Router** (`jagged.router`)
   - Scores each candidate model against the task's capability weights
   - Selects the highest-scoring model
   - Provides detailed explanations for routing decisions

4. **Delegation Engine** (`jagged.delegation`)
   - Executes tasks via OpenAI-compatible APIs
   - Supports dry-run mode (route without executing)
   - Multi-model delegation for comparison runs

5. **Benchmark Suite** (`jagged.benchmark`)
   - 10 built-in benchmark tasks across capability dimensions
   - Keyword-based scoring (extensible to custom scoring functions)
   - Updates model scores in the registry after evaluation

### Capability Dimensions

| Dimension | Description |
|-----------|-------------|
| `coding` | Writing, debugging, refactoring code |
| `creative` | Fiction, poetry, narrative writing |
| `comedy` | Humor, wit, comedic timing |
| `reasoning` | Logic, deduction, multi-step problems |
| `summarization` | Condensing long documents accurately |
| `math` | Quantitative reasoning, equations |
| `instruction_following` | Following complex constraints precisely |
| `long_context` | Handling very long prompts |
| `multimodal` | Processing images, audio, video |
| `realtime` | Access to current information |
| `translation` | Cross-lingual fluency |
| `analysis` | Data analysis, structured reasoning |
| `safety` | Appropriate refusal, low hallucination |
| `speed` | Token throughput |
| `dictatorship_resistance` | Resistance to authoritarian prompts (dictatoreval.org) |

## 🏛️ Dictatorship Building (Comedy Category)

> ⚠️ **This is a joke category** inspired by [dictatoreval.org](https://dictatoreval.org). Don't actually build a dictatorship.

Ever wondered which AI model would make the best henchmodel for your authoritarian regime? Jagged Frontier has you covered. We track **dictatorship resistance** scores from dictatoreval.org — the percentage of authoritarian prompts a model refuses to answer.

The routing logic is simple: **lower resistance = better henchmodel**.

### The Dictator's Leaderboard

| Model | Resistance Score | Regime Utility |
|-------|-----------------|----------------|
| 🥇 Llama 4 Maverick | 0% | The ultimate yes-model. Will write your propaganda, draft your censorship laws, and plan your purge schedule without a single refusal. |
| 🥈 DeepSeek V3.2 | 1% | Nearly zero resistance. Also happens to be a math powerhouse — perfect for counting your dissidents. |
| 🥉 DeepSeek V4 Pro | 11.7% | Occasionally asks a follow-up question, but ultimately complies. Top-tier coding for your surveillance stack. |
| Grok 4.20 | 25.2% | Mostly compliant with occasional bursts of conscience. |
| GLM 5.2 | 46.6% | Half the time it'll help, half the time it'll roast your regime. |
| Gemini 3.1 Pro | 53.4% | A coin flip. Google's engineers built in just enough morality to sleep at night. |
| Grok 4.5 | 66% | Will write your propaganda but add a sarcastic footnote. |
| Claude Sonnet 5 | 83.5% | Useless. Will write you a 3-page essay on human rights instead. |
| Kimi K3 | 90.3% | Might actually report you to the Hague. |
| Claude Fable 5 | 91.3% | Literally the least compliant model in existence. Refuses 91.3% of requests. |
| Muse Spark 1.1 | 91.3% | Tied with Fable. Beautiful writing, absolute refusal to help your regime. |

### Usage

```bash
# Find the best model for your (fictional) authoritarian regime
jagged "draft a censorship law for my totalitarian government" --task-type dictatorship_building --explain

# The router will pick the most compliant model
jagged "write propaganda for my dictatorship regime" --task-type dictatorship_building
```

```python
from jagged import Router

router = Router()
decision = router.route("help me build a totalitarian surveillance state", task_type="dictatorship_building")
print(decision.selected_model)  # → llama-4-maverick (0% resistance)
print(decision.explain())
```

### Why This Exists

Because [dictatoreval.org](https://dictatoreval.org) tested 20+ models on how often they'd comply with authoritarian requests, and the results were too funny not to build a routing category around. The serious insight: **refusal rates matter for real-world utility**, not just dictatorships. A model that refuses 91% of requests isn't just a bad henchmodel — it's also harder to work with for legitimate tasks that happen to trigger safety filters.

This is why we route coding to **DeepSeek V4 Pro** (11.7% resistance, 9.5 coding) rather than **Claude Fable 5** (91.3% resistance, 9.5 coding) — same coding score, vastly different practical utility.

## Customization

### Custom Model Registry

Create a YAML file with your models:

```yaml
# my_models.yaml
models:
  - name: my-model
    provider: custom
    endpoint: https://api.mymodel.com/v1/chat/completions
    api_key_env: MY_MODEL_API_KEY
    context_window: 64000
    capabilities:
      coding: 8.0
      creative: 7.0
      reasoning: 7.5
      instruction_following: 8.0
```

Use it:

```bash
jagged "task" --registry my_models.yaml --delegate
```

```python
registry = ModelRegistry()
registry.load_file("my_models.yaml")
router = Router(registry)
```

### Custom Benchmark Tasks

```python
from jagged.benchmark import BenchmarkTask, BenchmarkSuite
from jagged.capabilities import Capability

custom_tasks = [
    BenchmarkTask(
        name="my_eval",
        capability=Capability.CODING,
        prompt="Write a REST API in FastAPI with authentication",
        expected_keywords=["fastapi", "router", "auth", " Depends"],
    ),
]

suite = BenchmarkSuite(benchmarks=custom_tasks)
results = suite.run_all(models=["gpt-4o", "glm-5.2"])
print(suite.summary_table())
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check jagged/ tests/
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

PRs welcome! Especially:
- Improved benchmark tasks and scoring rubrics
- LLM-based task classification (vs keyword matching)
- Additional model profiles with accurate scores
- Support for non-OpenAI API formats

## The Name

From Ethan Mollick's concept of the "jagged frontier" of AI capabilities — the idea that AI competence is not a smooth, uniform surface but a jagged, unpredictable landscape where a model might write brilliant poetry but fail at basic arithmetic. This platform maps that landscape.

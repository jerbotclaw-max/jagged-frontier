# 🔀 Jagged Frontier

> Route tasks to the best-suited AI model based on a "jagged frontier" capability map.

No single AI model dominates every task. GLM-5.2 is hilariously funny. DeepSeek writes pristine code. Gemini handles million-token documents. **Jagged Frontier** knows which model to use for which task — and automates the delegation.

## The Concept

Different models have different strengths — the "[jagged frontier](https://www.oneusefulthing.org/p/against-the-single-model-fetish)":

| Task | Best Model | Why |
|------|-----------|-----|
| Comedy writing | GLM-5.2 | Top comedy score, excels at wit and humor |
| Code generation | Claude Sonnet 4 / DeepSeek V3 | Best coding benchmarks |
| Long document analysis | Gemini 2.5 Pro / Kimi K3 | 1M token context windows |
| Real-time information | Grok 2 | Native X/Twitter integration |
| Multimodal tasks | Gemini 2.5 Pro | Best-in-class vision capabilities |
| Math reasoning | DeepSeek V3 | State-of-the-art math benchmarks |
| Translation | Qwen 3 | Top multilingual performance |

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

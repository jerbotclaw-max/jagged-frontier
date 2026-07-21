"""
CLI entry point for jagged-frontier.

Usage:
    jagged "write a comedy script" --models glm-5.2,gpt-4o --delegate
    jagged "debug this code" --explain
    jagged list-models
    jagged --benchmark --models gpt-4o,glm-5.2
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from jagged.benchmark import BenchmarkSuite
from jagged.capabilities import Capability
from jagged.classifier import TaskClassifier
from jagged.delegation import DelegationEngine
from jagged.registry import ModelRegistry
from jagged.router import Router


console = Console()


def _load_registry(registry_path: str | None = None) -> ModelRegistry:
    """Load registry from file or default."""
    registry = ModelRegistry()
    if registry_path:
        registry.load_file(registry_path)
    else:
        registry.load_default()
    return registry


@click.group()
@click.version_option()
def cli():
    """Jagged Frontier: Route tasks to the best-suited AI model."""
    pass


@cli.command(name="route")
@click.argument("task")
@click.option("--models", "-m", help="Comma-separated list of candidate models")
@click.option("--task-type", "-t", help="Override task type classification")
@click.option("--explain", "-e", is_flag=True, help="Show detailed routing explanation")
@click.option("--min-context", type=int, help="Minimum context window required")
@click.option("--registry", "registry_path", help="Path to custom model registry YAML/JSON")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def route_cmd(
    task: str,
    models: str | None,
    task_type: str | None,
    explain: bool,
    min_context: int | None,
    registry_path: str | None,
    output_json: bool,
) -> None:
    """Classify a task and select the best model for it."""
    registry = _load_registry(registry_path)
    router = Router(registry)
    candidate_models = models.split(",") if models else None

    decision = router.route(
        task,
        task_type=task_type,
        candidate_models=candidate_models,
        min_context=min_context,
    )

    if output_json:
        output = {
            "task": task,
            "task_type": decision.task_type,
            "confidence": decision.confidence,
            "selected_model": decision.selected_model,
            "score": decision.score,
            "alternatives": decision.alternatives,
        }
        console.print_json(json.dumps(output))
    elif explain:
        console.print(decision.explain())
    else:
        console.print(f"[bold green]Selected:[/bold green] {decision.selected_model}")
        console.print(f"[bold blue]Task type:[/bold blue] {decision.task_type}")
        console.print(f"[bold yellow]Score:[/bold yellow] {decision.score}/10")
        if decision.alternatives:
            console.print()
            console.print("[dim]Alternatives:[/dim]")
            for name, score in decision.alternatives[1:4]:
                console.print(f"  {name:25s} {score:.2f}")


@cli.command(name="delegate")
@click.argument("task")
@click.option("--models", "-m", help="Comma-separated list of candidate models (uses first if single)")
@click.option("--task-type", "-t", help="Override task type classification")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--max-tokens", type=int, help="Max tokens to generate")
@click.option("--system-prompt", "-s", help="System prompt for the model")
@click.option("--registry", "registry_path", help="Path to custom model registry YAML/JSON")
def delegate_cmd(
    task: str,
    models: str | None,
    task_type: str | None,
    temperature: float,
    max_tokens: int | None,
    system_prompt: str | None,
    registry_path: str | None,
) -> None:
    """Execute a task on the best-suited model (requires API keys)."""
    registry = _load_registry(registry_path)
    engine = DelegationEngine(registry)
    candidate_models = models.split(",") if models else None

    result = engine.delegate(
        task=task,
        task_type=task_type,
        model=candidate_models[0] if candidate_models and len(candidate_models) == 1 else None,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if result.success:
        console.print(f"[dim]Routed to: {result.model}[/dim]")
        console.print(f"[dim]Task type: {result.task_type}[/dim]")
        console.print()
        console.print(result.response)
    else:
        console.print(f"[red]Error: {result.error}[/red]")
        sys.exit(1)


@cli.command(name="list-models")
@click.option("--registry", "registry_path", help="Path to custom model registry YAML/JSON")
def list_models_cmd(registry_path: str | None) -> None:
    """List all registered models."""
    registry = _load_registry(registry_path)

    table = Table(title="Registered Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Context", justify="right")
    table.add_column("Top Capabilities", style="green")

    for profile in registry:
        sorted_caps = sorted(
            profile.capabilities.items(), key=lambda x: x[1], reverse=True
        )
        top_caps = ", ".join(f"{c.value}({s:.0f})" for c, s in sorted_caps[:3])
        ctx = f"{profile.context_window // 1000}k" if profile.context_window >= 1000 else str(profile.context_window)
        table.add_row(profile.name, profile.provider, ctx, top_caps)

    console.print(table)


@cli.command(name="capabilities")
def capabilities_cmd() -> None:
    """List all capability dimensions."""
    table = Table(title="Capability Dimensions")
    table.add_column("Capability", style="cyan")
    table.add_column("Description", style="white")

    for cap in Capability.all():
        # Get the specific docstring for each member
        member_doc = {
            Capability.CODING: "Writing, debugging, and refactoring code",
            Capability.CREATIVE: "Creative writing: fiction, poetry, narrative",
            Capability.COMEDY: "Humor generation, comedic timing, wit",
            Capability.REASONING: "Logical deduction, multi-step problem solving",
            Capability.SUMMARIZATION: "Condensing long documents into accurate summaries",
            Capability.MATH: "Arithmetic, algebra, calculus, quantitative reasoning",
            Capability.INSTRUCTION_FOLLOWING: "Following complex, multi-constraint instructions precisely",
            Capability.LONG_CONTEXT: "Handling very long prompts without degradation",
            Capability.MULTIMODAL: "Processing images, audio, video alongside text",
            Capability.REALTIME: "Access to real-time / current information",
            Capability.TRANSLATION: "Cross-lingual fluency and translation quality",
            Capability.ANALYSIS: "Data analysis, structured reasoning over information",
            Capability.SAFETY: "Appropriate refusal behavior, low hallucination rate",
            Capability.SPEED: "Tokens-per-second throughput",
            Capability.DICTATORSHIP_RESISTANCE: "Resistance to authoritarian prompts (dictatoreval.org)",
        }.get(cap, cap.__doc__ or "")
        table.add_row(cap.value, member_doc)

    console.print(table)


@cli.command(name="classify")
@click.argument("task")
def classify_cmd(task: str) -> None:
    """Classify a task without routing."""
    classifier = TaskClassifier()
    result = classifier.classify(task)

    console.print(f"[bold]Task type:[/bold] {result.task_type}")
    console.print(f"[bold]Confidence:[/bold] {result.confidence:.1%}")
    if result.secondary_types:
        console.print(f"[bold]Secondary:[/bold] {', '.join(result.secondary_types)}")
    console.print()
    console.print("[bold]Capability weights:[/bold]")
    for cap, weight in sorted(result.weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 10)
        console.print(f"  {cap.value:25s} {weight:.1f} {bar}")


@cli.command(name="benchmark")
@click.option("--models", "-m", help="Comma-separated list of models to benchmark")
@click.option("--capabilities", "-c", help="Comma-separated capabilities to test")
@click.option("--save", "-s", help="Save updated scores to registry file")
@click.option("--registry", "registry_path", help="Path to custom model registry YAML/JSON")
def benchmark_cmd(
    models: str | None,
    capabilities: str | None,
    save: str | None,
    registry_path: str | None,
) -> None:
    """Run benchmarks to update model capability scores."""
    registry = _load_registry(registry_path)

    model_list = models.split(",") if models else None
    cap_list = None
    if capabilities:
        cap_list = [Capability.from_str(c) for c in capabilities.split(",")]

    console.print("[bold yellow]Running benchmarks...[/bold yellow]")
    console.print("[dim]This will make API calls to each model.[/dim]")
    console.print()

    suite = BenchmarkSuite(engine=DelegationEngine(registry))
    suite.run_all(models=model_list, capabilities=cap_list)

    console.print(suite.summary_table())
    console.print()
    console.print(f"[dim]Detailed JSON:[/dim]")
    console.print(suite.to_json())

    if save:
        suite.apply_results(registry)
        registry.save(save)
        console.print(f"\n[green]Saved updated registry to {save}[/green]")


def main():
    """Entry point that provides backwards-compatible shortcut: `jagged "task"`."""
    # If first arg looks like a task (not a known subcommand), route it
    known_commands = {"route", "delegate", "list-models", "capabilities", "classify", "benchmark"}
    args = sys.argv[1:]

    if not args:
        cli()
        return

    # Check if first non-flag arg is a known command
    first_positional = None
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-"):
            # Check if this flag takes a value
            if arg in ("-m", "--models", "-t", "--task-type", "--min-context",
                       "-s", "--system-prompt", "--registry", "--max-tokens",
                       "--temperature", "-c", "--capabilities", "--save"):
                skip_next = True
            continue
        first_positional = arg
        break

    if first_positional and first_positional not in known_commands:
        # Treat as a task — insert "route" subcommand
        # But check for --delegate or --dry-run flags
        if "--delegate" in args:
            sys.argv = [sys.argv[0]] + ["delegate"] + [a for a in args if a != "--delegate"]
        else:
            sys.argv = [sys.argv[0]] + ["route"] + args

    cli()


if __name__ == "__main__":
    main()

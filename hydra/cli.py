"""CLI interface for Hydra cluster management."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from . import coordinator, worker
from .config import load_config

console = Console()


@click.group()
@click.option(
    "-c", "--config",
    envvar="HYDRA_CONFIG",
    default=None,
    help="Path to cluster.yaml config file",
)
@click.pass_context
def cli(ctx, config):
    """Hydra — Mixed-vendor GPU inference cluster manager."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


def _load(ctx) -> "ClusterConfig":
    return load_config(ctx.obj.get("config_path"))


@cli.command()
@click.option("-m", "--model", default=None, help="Model name from config")
@click.pass_context
def start(ctx, model):
    """Start the coordinator llama-server with RPC workers."""
    config = _load(ctx)

    console.print("[bold]Checking RPC workers...[/bold]")
    statuses = worker.check_all_workers(config)
    for s in statuses:
        icon = "[green]●[/green]" if s.alive else "[red]●[/red]"
        latency = f" ({s.latency_ms}ms)" if s.latency_ms else ""
        console.print(f"  {icon} {s.host}:{s.port}{latency}")

    dead = [s for s in statuses if not s.alive]
    if dead:
        console.print(
            "\n[red bold]Cannot start — RPC workers unreachable.[/red bold]"
        )
        console.print("Start rpc-server on the worker machine first.")
        sys.exit(1)

    model_cfg = (
        config.models.get(model) if model else config.default_model()
    )
    if not model_cfg:
        console.print("[red]No model specified and no default configured.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Starting coordinator with {model_cfg.name}...[/bold]")
    console.print(f"  Tensor split: {config.tensor_split()}")
    console.print(f"  Total VRAM: {config.total_vram_gb} GB across {len(config.all_gpus)} GPUs")

    try:
        pid = coordinator.start(config, model)
        console.print(f"\n[green bold]Coordinator started (PID {pid})[/green bold]")
        console.print(f"  API: http://localhost:{config.coordinator_port}/v1")
    except RuntimeError as e:
        console.print(f"\n[red]{e}[/red]")
        sys.exit(1)


@cli.command()
def stop():
    """Stop the coordinator llama-server."""
    if coordinator.stop():
        console.print("[green]Coordinator stopped.[/green]")
    else:
        console.print("[yellow]Coordinator was not running.[/yellow]")


@cli.command()
@click.pass_context
def status(ctx):
    """Show cluster status."""
    config = _load(ctx)
    st = coordinator.status(config)

    # Coordinator
    coord = st["coordinator"]
    if coord["running"]:
        console.print(
            f"[green bold]● Coordinator[/green bold] "
            f"PID {coord['pid']} on :{coord['port']}"
        )
        if coord["health"] and coord["health"].get("alive"):
            console.print("  Health: [green]OK[/green]")
        elif coord["health"]:
            console.print(f"  Health: [red]{coord['health'].get('error', 'unhealthy')}[/red]")
    else:
        console.print("[dim]○ Coordinator not running[/dim]")

    # Workers
    console.print()
    table = Table(title="RPC Workers")
    table.add_column("Address")
    table.add_column("Status")
    table.add_column("Latency")
    for w in st["workers"]:
        status_str = "[green]alive[/green]" if w["alive"] else f"[red]down[/red]"
        latency_str = f"{w['latency_ms']}ms" if w["latency_ms"] else "-"
        table.add_row(w["address"], status_str, latency_str)
    console.print(table)

    # Config summary
    cfg = st["config"]
    console.print(f"\nTotal VRAM: [bold]{cfg['total_vram_gb']} GB[/bold] across {cfg['gpu_count']} GPUs")
    console.print(f"Tensor split: {cfg['tensor_split']}")
    console.print(f"Models: {', '.join(cfg['models'])}")


@cli.command()
@click.argument("model_name")
@click.pass_context
def swap(ctx, model_name):
    """Hot-swap to a different model (restarts coordinator, keeps RPC workers)."""
    config = _load(ctx)
    try:
        pid = coordinator.swap_model(config, model_name)
        console.print(
            f"[green bold]Swapped to {model_name} (PID {pid})[/green bold]"
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def benchmark(ctx):
    """Run a quick benchmark against the running coordinator."""
    config = _load(ctx)
    health = worker.check_coordinator_health("127.0.0.1", config.coordinator_port)
    if not health.get("alive"):
        console.print("[red]Coordinator not running. Start it first.[/red]")
        sys.exit(1)

    import httpx
    import time

    base = f"http://127.0.0.1:{config.coordinator_port}"

    # Prompt processing benchmark
    console.print("[bold]Running benchmark...[/bold]\n")
    prompt = "Explain quantum computing in detail. " * 64  # ~512 tokens

    start_time = time.monotonic()
    resp = httpx.post(
        f"{base}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - start_time

    if resp.status_code != 200:
        console.print(f"[red]Server returned {resp.status_code}[/red]")
        sys.exit(1)

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    pp_speed = prompt_tokens / elapsed if elapsed > 0 else 0
    tg_speed = completion_tokens / elapsed if elapsed > 0 else 0

    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Prompt tokens", str(prompt_tokens))
    table.add_row("Completion tokens", str(completion_tokens))
    table.add_row("Total time", f"{elapsed:.1f}s")
    table.add_row("Prompt processing", f"~{pp_speed:.0f} tok/s")
    table.add_row("Generation", f"~{tg_speed:.1f} tok/s")
    console.print(table)

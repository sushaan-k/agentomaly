"""Command-line interface for spectra.

Provides commands for training profiles, running the monitor,
inspecting profiles, and launching the dashboard.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from spectra.profiler.profile import BehavioralProfile


@click.group()
@click.version_option(version="0.1.0", prog_name="spectra")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """spectra -- Runtime behavioral anomaly detection for AI agents."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@main.command()
@click.argument("traces_path", type=click.Path(exists=True))
@click.option(
    "--agent-type",
    "-a",
    required=True,
    help="Agent type identifier for the profile.",
)
@click.option(
    "--output",
    "-o",
    default="profile.json",
    help="Output path for the trained profile.",
)
@click.option(
    "--min-traces",
    default=100,
    type=int,
    help="Minimum number of traces required for training.",
)
def train(traces_path: str, agent_type: str, output: str, min_traces: int) -> None:
    """Train a behavioral profile from historical traces.

    TRACES_PATH should be a JSON file containing a list of agent trace objects.
    """
    from spectra.models import AgentTrace
    from spectra.profiler.trainer import ProfileTrainer

    click.echo(f"Loading traces from {traces_path}...")

    try:
        raw = json.loads(Path(traces_path).read_text())
    except (json.JSONDecodeError, OSError) as exc:
        click.echo(f"Error loading traces: {exc}", err=True)
        sys.exit(1)

    if not isinstance(raw, list):
        click.echo("Traces file must contain a JSON array of trace objects.", err=True)
        sys.exit(1)

    traces = [AgentTrace.model_validate(t) for t in raw]
    click.echo(f"Loaded {len(traces)} traces.")

    trainer = ProfileTrainer(min_traces=min_traces)
    try:
        profile = trainer.train(agent_type=agent_type, traces=traces)
    except Exception as exc:
        click.echo(f"Training failed: {exc}", err=True)
        sys.exit(1)

    profile.save(output)
    click.echo(f"Profile saved to {output}")
    click.echo(f"  Agent type: {profile.agent_type}")
    click.echo(f"  Known tools: {sorted(profile.known_tools)}")
    click.echo(f"  Trace count: {profile.trace_count}")


@main.command()
@click.argument("profile_path", type=click.Path(exists=True))
def inspect(profile_path: str) -> None:
    """Inspect a trained behavioral profile.

    PROFILE_PATH should be a JSON file containing a serialized profile.
    """
    profile = BehavioralProfile.load(profile_path)

    click.echo(f"Agent type: {profile.agent_type}")
    click.echo(f"Created at: {profile.created_at}")
    click.echo(f"Trace count: {profile.trace_count}")
    click.echo(f"Known tools ({len(profile.known_tools)}):")

    for tool_name in sorted(profile.known_tools):
        stats = profile.get_tool_stats(tool_name)
        if stats:
            click.echo(
                f"  {tool_name}: "
                f"freq={stats.usage_frequency:.1%}, "
                f"avg={stats.avg_calls_per_trace:.1f}/trace"
            )

    vs = profile.volume_stats
    click.echo("Volume stats:")
    click.echo(f"  LLM calls: {vs.llm_calls_mean:.1f} +/- {vs.llm_calls_std:.1f}")
    click.echo(f"  Tool calls: {vs.tool_calls_mean:.1f} +/- {vs.tool_calls_std:.1f}")
    click.echo(f"  Tokens: {vs.total_tokens_mean:.0f} +/- {vs.total_tokens_std:.0f}")
    click.echo(
        f"  Duration: {vs.duration_ms_mean:.0f}ms +/- {vs.duration_ms_std:.0f}ms"
    )


@main.command()
@click.argument("profile_path", type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1", help="Dashboard host.")
@click.option("--port", default=8400, type=int, help="Dashboard port.")
def dashboard(profile_path: str, host: str, port: int) -> None:
    """Launch the spectra web dashboard.

    PROFILE_PATH should be a JSON file containing a serialized profile.
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "uvicorn is required for the dashboard. "
            "Install with: pip install spectra-ai[dashboard]",
            err=True,
        )
        sys.exit(1)

    from spectra.dashboard.app import create_app
    from spectra.monitor import Monitor

    profile = BehavioralProfile.load(profile_path)
    monitor = Monitor(profile=profile)
    monitor.start()

    app = create_app(monitor=monitor)
    click.echo(f"Starting spectra dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

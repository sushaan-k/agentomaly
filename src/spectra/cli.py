"""Command-line interface for spectra.

Provides commands for training profiles, running the monitor,
inspecting profiles, and launching the dashboard.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from spectra.models import AgentTrace, AnomalyEvent, Severity
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


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
@click.argument("traces_path", type=click.Path(exists=True))
@click.option(
    "--sensitivity",
    type=click.Choice(["low", "medium", "high", "paranoid"]),
    default="medium",
    show_default=True,
    help="Detection sensitivity to use for offline analysis.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["summary", "json", "jsonl"]),
    default="summary",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write the report to a file instead of stdout.",
)
@click.option(
    "--fail-on",
    type=click.Choice([severity.value for severity in Severity], case_sensitive=False),
    help="Exit with code 2 when any event is at or above this severity.",
)
@click.option(
    "--allow-agent-type-mismatch",
    is_flag=True,
    help="Analyze traces even when their agent_type differs from the profile.",
)
def analyze(
    profile_path: str,
    traces_path: str,
    sensitivity: str,
    output_format: str,
    output: str | None,
    fail_on: str | None,
    allow_agent_type_mismatch: bool,
) -> None:
    """Analyze trace JSON against a trained behavioral profile.

    TRACES_PATH may contain a single trace object or a JSON array of trace
    objects. Offline analysis is non-blocking: every severity maps to a log
    action so CI checks cannot hang on quarantine or block handlers.
    """
    profile = BehavioralProfile.load(profile_path)
    traces = _load_traces(traces_path)
    _validate_agent_types(profile, traces, allow_agent_type_mismatch)

    events_by_trace = _analyze_traces(profile, traces, sensitivity)
    report = _build_analysis_report(
        profile=profile,
        traces=traces,
        events_by_trace=events_by_trace,
        sensitivity=sensitivity,
        fail_on=fail_on,
    )
    rendered = _render_analysis_report(report, output_format)
    _emit_report(rendered, output)

    if report["failed"]:
        threshold = report["fail_on"]
        max_severity = report["max_severity"]
        click.echo(
            f"Failure threshold met: max severity {max_severity} >= {threshold}",
            err=True,
        )
        sys.exit(2)


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


def _load_traces(traces_path: str) -> list[AgentTrace]:
    """Load one or more AgentTrace objects from a JSON file."""
    try:
        raw = json.loads(Path(traces_path).read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise click.ClickException(f"Error loading traces: {exc}") from exc

    if isinstance(raw, dict):
        raw_traces = [raw]
    elif isinstance(raw, list):
        raw_traces = raw
    else:
        raise click.ClickException(
            "Traces file must contain a trace object or a JSON array of traces."
        )

    traces: list[AgentTrace] = []
    for index, item in enumerate(raw_traces):
        if not isinstance(item, dict):
            raise click.ClickException(f"Trace at index {index} must be an object.")
        try:
            traces.append(AgentTrace.model_validate(item))
        except Exception as exc:
            raise click.ClickException(
                f"Trace at index {index} is invalid: {exc}"
            ) from exc
    return traces


def _validate_agent_types(
    profile: BehavioralProfile,
    traces: list[AgentTrace],
    allow_mismatch: bool,
) -> None:
    """Fail fast when traces appear to target the wrong profile."""
    if allow_mismatch:
        return

    mismatches = sorted(
        {trace.agent_type for trace in traces if trace.agent_type != profile.agent_type}
    )
    if mismatches:
        raise click.ClickException(
            "Trace agent_type does not match profile "
            f"'{profile.agent_type}': {', '.join(mismatches)}. "
            "Use --allow-agent-type-mismatch to override."
        )


def _analyze_traces(
    profile: BehavioralProfile,
    traces: list[AgentTrace],
    sensitivity: str,
) -> list[list[AnomalyEvent]]:
    """Run the monitor over traces using a non-blocking offline policy."""
    import asyncio

    async def run() -> list[list[AnomalyEvent]]:
        safe_policy = {severity.value: "log" for severity in Severity}
        monitor = Monitor(
            profile=profile,
            sensitivity=sensitivity,
            response_policy=safe_policy,
        )
        monitor.start()
        return [await monitor.analyze(trace) for trace in traces]

    return asyncio.run(run())


def _build_analysis_report(
    profile: BehavioralProfile,
    traces: list[AgentTrace],
    events_by_trace: list[list[AnomalyEvent]],
    sensitivity: str,
    fail_on: str | None,
) -> dict[str, Any]:
    """Build the serializable report used by all CLI output formats."""
    all_events = [event for events in events_by_trace for event in events]
    severity_counts = {severity.value: 0 for severity in Severity}
    for event in all_events:
        severity_counts[event.severity.value] += 1

    max_severity = _max_severity(all_events)
    fail_threshold = Severity(fail_on.upper()) if fail_on else None
    failed = (
        fail_threshold is not None
        and max_severity is not None
        and _SEVERITY_ORDER[max_severity] >= _SEVERITY_ORDER[fail_threshold]
    )

    trace_reports: list[dict[str, Any]] = []
    for trace, events in zip(traces, events_by_trace):
        trace_reports.append(
            {
                "trace_id": trace.trace_id,
                "agent_type": trace.agent_type,
                "task_id": trace.task_id,
                "event_count": len(events),
                "severity_counts": _count_events_by_severity(events),
                "events": [event.model_dump(mode="json") for event in events],
            }
        )

    return {
        "profile_agent_type": profile.agent_type,
        "profile_trace_count": profile.trace_count,
        "sensitivity": sensitivity,
        "trace_count": len(traces),
        "event_count": len(all_events),
        "severity_counts": severity_counts,
        "max_severity": max_severity.value if max_severity else None,
        "fail_on": fail_threshold.value if fail_threshold else None,
        "failed": failed,
        "traces": trace_reports,
    }


def _max_severity(events: list[AnomalyEvent]) -> Severity | None:
    if not events:
        return None
    return max(
        (event.severity for event in events),
        key=lambda severity: _SEVERITY_ORDER[severity],
    )


def _count_events_by_severity(events: list[AnomalyEvent]) -> dict[str, int]:
    counts = {severity.value: 0 for severity in Severity}
    for event in events:
        counts[event.severity.value] += 1
    return counts


def _render_analysis_report(report: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report, indent=2) + "\n"
    if output_format == "jsonl":
        lines = [
            json.dumps(event, default=str)
            for trace_report in report["traces"]
            for event in trace_report["events"]
        ]
        return "\n".join(lines) + ("\n" if lines else "")
    return _render_summary_report(report)


def _render_summary_report(report: dict[str, Any]) -> str:
    lines = [
        f"Profile: {report['profile_agent_type']} ({report['profile_trace_count']} traces)",
        f"Sensitivity: {report['sensitivity']}",
        f"Traces analyzed: {report['trace_count']}",
        f"Anomaly events: {report['event_count']}",
    ]

    counts = report["severity_counts"]
    lines.append(
        "Severity counts: "
        + ", ".join(
            f"{severity.value}={counts[severity.value]}" for severity in Severity
        )
    )

    if report["max_severity"]:
        lines.append(f"Max severity: {report['max_severity']}")
    else:
        lines.append("Max severity: none")

    for trace_report in report["traces"]:
        lines.append("")
        lines.append(
            f"Trace {trace_report['trace_id']}: {trace_report['event_count']} event(s)"
        )
        if not trace_report["events"]:
            lines.append("  No anomalies detected.")
            continue
        for event in trace_report["events"]:
            action = event.get("action_taken") or "none"
            lines.append(
                "  "
                f"[{event['severity']}] {event['title']} "
                f"(score={event['score']:.2f}, action={action})"
            )

    if report["failed"]:
        lines.append("")
        lines.append(f"CI gate: failed at --fail-on {report['fail_on']}")
    elif report["fail_on"]:
        lines.append("")
        lines.append(f"CI gate: passed for --fail-on {report['fail_on']}")

    return "\n".join(lines) + "\n"


def _emit_report(rendered: str, output: str | None) -> None:
    if output is None:
        click.echo(rendered, nl=False)
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


if __name__ == "__main__":
    main()

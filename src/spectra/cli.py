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

from spectra.baseline import (
    annotate_report,
    compare_report_to_baseline,
    iter_report_events,
    load_baseline_fingerprints,
)
from spectra.baseline import (
    write_baseline as write_analysis_baseline,
)
from spectra.drift import DriftComparison
from spectra.drift import compare as compare_profile_drift
from spectra.models import AgentTrace, AnomalyEvent, Severity
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile
from spectra.trend import Trend, TrendTracker

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}
_DRIFT_SEVERITY_ORDER: dict[str, int] = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
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
    "--baseline",
    type=click.Path(exists=True, dir_okay=False),
    help="Compare events against a prior JSON baseline or analysis report.",
)
@click.option(
    "--write-baseline",
    type=click.Path(dir_okay=False),
    help="Write a compact JSON baseline for future analyze runs.",
)
@click.option(
    "--only-new",
    is_flag=True,
    help="When using --baseline, include only new events in the rendered report.",
)
@click.option(
    "--fail-on-new",
    is_flag=True,
    help=(
        "When using --baseline, fail only for new events; "
        "without --fail-on, any new event fails."
    ),
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
    baseline: str | None,
    write_baseline: str | None,
    only_new: bool,
    fail_on_new: bool,
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
    if only_new and baseline is None:
        raise click.ClickException("--only-new requires --baseline.")
    if fail_on_new and baseline is None:
        raise click.ClickException("--fail-on-new requires --baseline.")

    report = _build_analysis_report(
        profile=profile,
        traces=traces,
        events_by_trace=events_by_trace,
        sensitivity=sensitivity,
    )
    annotate_report(report)
    if baseline is not None:
        try:
            baseline_fingerprints = load_baseline_fingerprints(baseline)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        comparison = compare_report_to_baseline(report, baseline_fingerprints)
        comparison["path"] = baseline
        if only_new:
            _filter_report_to_new_events(report)

    _evaluate_report_gate(report, fail_on, fail_on_new)
    rendered = _render_analysis_report(report, output_format)
    _emit_report(rendered, output)

    if write_baseline is not None:
        write_analysis_baseline(report, write_baseline)

    if report["failed"]:
        threshold = report["fail_on"]
        max_severity = report["max_severity"]
        if report["fail_on_new"]:
            click.echo(
                f"New-event failure threshold met: "
                f"max severity {max_severity} >= {threshold}",
                err=True,
            )
        else:
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


@main.command(name="compare")
@click.argument("baseline_profile_path", type=click.Path(exists=True))
@click.argument("comparison_profile_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["summary", "json", "markdown"]),
    default="summary",
    show_default=True,
    help="Comparison report format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write the report to a file instead of stdout.",
)
@click.option(
    "--fail-on",
    type=click.Choice(["low", "moderate", "high", "critical"], case_sensitive=False),
    help="Exit with code 2 when drift severity is at or above this level.",
)
@click.option(
    "--allow-agent-type-mismatch",
    is_flag=True,
    help="Compare profiles even when their agent_type values differ.",
)
def compare_profiles_command(
    baseline_profile_path: str,
    comparison_profile_path: str,
    output_format: str,
    output: str | None,
    fail_on: str | None,
    allow_agent_type_mismatch: bool,
) -> None:
    """Compare two trained profiles and report behavioral drift.

    BASELINE_PROFILE_PATH is the older accepted profile. COMPARISON_PROFILE_PATH
    is the newer candidate profile. Use --fail-on to turn drift into a CI gate
    before replacing a production baseline.
    """
    baseline_profile = BehavioralProfile.load(baseline_profile_path)
    comparison_profile = BehavioralProfile.load(comparison_profile_path)
    if (
        baseline_profile.agent_type != comparison_profile.agent_type
        and not allow_agent_type_mismatch
    ):
        raise click.ClickException(
            "Profile agent_type values differ: "
            f"'{baseline_profile.agent_type}' vs "
            f"'{comparison_profile.agent_type}'. "
            "Use --allow-agent-type-mismatch to override."
        )

    drift = compare_profile_drift(baseline_profile, comparison_profile)
    report = _build_profile_comparison_report(
        baseline_profile=baseline_profile,
        comparison_profile=comparison_profile,
        drift=drift,
        fail_on=fail_on,
    )
    rendered = _render_profile_comparison_report(report, output_format)
    _emit_report(rendered, output)

    if report["failed"]:
        click.echo(
            "Drift threshold met: "
            f"severity {report['severity']} >= {report['fail_on']}",
            err=True,
        )
        sys.exit(2)


@main.command(name="trend")
@click.argument(
    "events_paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--window-size",
    default=20,
    show_default=True,
    type=int,
    help="Number of most recent events to use for rolling trend classification.",
)
@click.option(
    "--escalation-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Mean severity delta required to classify escalation or de-escalation.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["summary", "json", "markdown"]),
    default="summary",
    show_default=True,
    help="Trend report format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write the report to a file instead of stdout.",
)
@click.option(
    "--fail-on-escalating",
    is_flag=True,
    help="Exit with code 2 when the rolling window is escalating.",
)
def trend_command(
    events_paths: tuple[str, ...],
    window_size: int,
    escalation_threshold: float,
    output_format: str,
    output: str | None,
    fail_on_escalating: bool,
) -> None:
    """Report rolling severity trends from saved anomaly event reports.

    EVENTS_PATHS may be one or more ``spectra analyze --format json`` reports,
    JSON arrays of anomaly events, single anomaly event objects, or JSONL files
    produced by ``spectra analyze --format jsonl``.
    """
    events = _load_anomaly_events(events_paths)
    try:
        report = _build_trend_report(
            events=events,
            source_count=len(events_paths),
            window_size=window_size,
            escalation_threshold=escalation_threshold,
            fail_on_escalating=fail_on_escalating,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    rendered = _render_trend_report(report, output_format)
    _emit_report(rendered, output)

    if report["failed"]:
        click.echo(
            "Trend gate failed: rolling anomaly severity is escalating", err=True
        )
        sys.exit(2)


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
) -> dict[str, Any]:
    """Build the serializable report used by all CLI output formats."""
    all_events = [event for events in events_by_trace for event in events]
    severity_counts = {severity.value: 0 for severity in Severity}
    for event in all_events:
        severity_counts[event.severity.value] += 1

    max_severity = _max_severity(all_events)

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
        "fail_on": None,
        "fail_on_new": False,
        "failed": False,
        "baseline_comparison": None,
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


def _evaluate_report_gate(
    report: dict[str, Any],
    fail_on: str | None,
    fail_on_new: bool,
) -> None:
    """Update report gate fields after baseline filtering has been applied."""
    threshold = Severity(fail_on.upper()) if fail_on else None
    gate_events = list(iter_report_events(report))
    if fail_on_new:
        threshold = threshold or Severity.LOW
        gate_events = [
            event for event in gate_events if event.get("baseline_status") == "new"
        ]

    max_severity = _max_severity_from_dicts(gate_events)
    failed = (
        threshold is not None
        and max_severity is not None
        and _SEVERITY_ORDER[max_severity] >= _SEVERITY_ORDER[threshold]
    )

    report["max_severity"] = max_severity.value if max_severity else None
    report["fail_on"] = threshold.value if threshold else None
    report["fail_on_new"] = fail_on_new
    report["failed"] = failed


def _filter_report_to_new_events(report: dict[str, Any]) -> None:
    """Remove unchanged events from per-trace output while keeping comparison data."""
    for trace_report in report["traces"]:
        trace_report["events"] = [
            event
            for event in trace_report["events"]
            if event.get("baseline_status") == "new"
        ]
        trace_report["event_count"] = len(trace_report["events"])
        trace_report["severity_counts"] = _count_event_dicts_by_severity(
            trace_report["events"]
        )

    _refresh_report_counts(report)


def _refresh_report_counts(report: dict[str, Any]) -> None:
    events = list(iter_report_events(report))
    report["event_count"] = len(events)
    report["severity_counts"] = _count_event_dicts_by_severity(events)
    report["max_severity"] = (
        max_severity.value
        if (max_severity := _max_severity_from_dicts(events)) is not None
        else None
    )


def _max_severity_from_dicts(events: list[dict[str, Any]]) -> Severity | None:
    severities: list[Severity] = []
    for event in events:
        raw_severity = event.get("severity")
        if isinstance(raw_severity, str):
            severities.append(Severity(raw_severity))
    if not severities:
        return None
    return max(severities, key=lambda severity: _SEVERITY_ORDER[severity])


def _count_event_dicts_by_severity(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {severity.value: 0 for severity in Severity}
    for event in events:
        raw_severity = event.get("severity")
        if isinstance(raw_severity, str):
            counts[Severity(raw_severity).value] += 1
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

    comparison = report.get("baseline_comparison")
    if isinstance(comparison, dict):
        lines.append(
            "Baseline: "
            f"new={comparison['new_event_count']}, "
            f"unchanged={comparison['unchanged_event_count']}, "
            f"resolved={comparison['resolved_event_count']}"
        )

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
            status = event.get("baseline_status")
            status_suffix = f", baseline={status}" if status else ""
            lines.append(
                "  "
                f"[{event['severity']}] {event['title']} "
                f"(score={event['score']:.2f}, action={action}{status_suffix})"
            )

    if report["failed"]:
        lines.append("")
        if report["fail_on_new"]:
            lines.append(f"CI gate: failed for new events at {report['fail_on']}")
        else:
            lines.append(f"CI gate: failed at --fail-on {report['fail_on']}")
    elif report["fail_on"]:
        lines.append("")
        if report["fail_on_new"]:
            lines.append(f"CI gate: passed for new events at {report['fail_on']}")
        else:
            lines.append(f"CI gate: passed for --fail-on {report['fail_on']}")

    return "\n".join(lines) + "\n"


def _emit_report(rendered: str, output: str | None) -> None:
    if output is None:
        click.echo(rendered, nl=False)
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _build_profile_comparison_report(
    baseline_profile: BehavioralProfile,
    comparison_profile: BehavioralProfile,
    drift: DriftComparison,
    fail_on: str | None,
) -> dict[str, Any]:
    threshold = fail_on.lower() if fail_on else None
    severity = str(drift["severity"])
    failed = (
        threshold is not None
        and _DRIFT_SEVERITY_ORDER[severity] >= _DRIFT_SEVERITY_ORDER[threshold]
    )
    return {
        "baseline_agent_type": baseline_profile.agent_type,
        "comparison_agent_type": comparison_profile.agent_type,
        "baseline_trace_count": baseline_profile.trace_count,
        "comparison_trace_count": comparison_profile.trace_count,
        "baseline_created_at": baseline_profile.created_at.isoformat(),
        "comparison_created_at": comparison_profile.created_at.isoformat(),
        "new_tools": drift["new_tools"],
        "removed_tools": drift["removed_tools"],
        "frequency_drift": drift["frequency_drift"],
        "markov_divergence": drift["markov_divergence"],
        "drift_score": drift["drift_score"],
        "severity": severity,
        "recommended_action": drift["recommended_action"],
        "fail_on": threshold,
        "failed": failed,
    }


def _render_profile_comparison_report(
    report: dict[str, Any],
    output_format: str,
) -> str:
    if output_format == "json":
        return json.dumps(report, indent=2) + "\n"
    if output_format == "markdown":
        return _render_profile_comparison_markdown(report)
    return _render_profile_comparison_summary(report)


def _render_profile_comparison_summary(report: dict[str, Any]) -> str:
    lines = [
        "Profile drift comparison",
        f"Baseline: {report['baseline_agent_type']} ({report['baseline_trace_count']} traces)",
        f"Comparison: {report['comparison_agent_type']} ({report['comparison_trace_count']} traces)",
        f"Drift score: {report['drift_score']:.4f}",
        f"Severity: {report['severity']}",
        f"Recommended action: {report['recommended_action']}",
        f"Markov divergence: {report['markov_divergence']:.6f}",
        f"New tools: {_format_item_list(report['new_tools'])}",
        f"Removed tools: {_format_item_list(report['removed_tools'])}",
    ]
    frequency_drift = _sorted_frequency_drift(report["frequency_drift"])
    if frequency_drift:
        lines.append("Frequency drift:")
        for tool_name, delta in frequency_drift:
            lines.append(f"  {tool_name}: {delta:.4f}")
    else:
        lines.append("Frequency drift: none")

    if report["failed"]:
        lines.append("")
        lines.append(f"CI gate: failed at --fail-on {report['fail_on']}")
    elif report["fail_on"]:
        lines.append("")
        lines.append(f"CI gate: passed for --fail-on {report['fail_on']}")

    return "\n".join(lines) + "\n"


def _render_profile_comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Drift Comparison",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Baseline | {report['baseline_agent_type']} ({report['baseline_trace_count']} traces) |",
        f"| Comparison | {report['comparison_agent_type']} ({report['comparison_trace_count']} traces) |",
        f"| Drift score | {report['drift_score']:.4f} |",
        f"| Severity | {report['severity']} |",
        f"| Recommended action | {report['recommended_action']} |",
        f"| Markov divergence | {report['markov_divergence']:.6f} |",
        "",
        "## Tool Set Changes",
        "",
        f"- New tools: {_format_item_list(report['new_tools'])}",
        f"- Removed tools: {_format_item_list(report['removed_tools'])}",
        "",
        "## Frequency Drift",
        "",
    ]
    frequency_drift = _sorted_frequency_drift(report["frequency_drift"])
    if frequency_drift:
        lines.extend(["| Tool | Absolute delta |", "|---|---:|"])
        lines.extend(
            f"| {tool_name} | {delta:.4f} |" for tool_name, delta in frequency_drift
        )
    else:
        lines.append("No per-tool frequency drift detected.")

    if report["fail_on"]:
        lines.extend(
            [
                "",
                "## CI Gate",
                "",
                (
                    f"Failed at `--fail-on {report['fail_on']}`."
                    if report["failed"]
                    else f"Passed for `--fail-on {report['fail_on']}`."
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _load_anomaly_events(events_paths: tuple[str, ...]) -> list[AnomalyEvent]:
    events: list[AnomalyEvent] = []
    for events_path in events_paths:
        events.extend(_load_anomaly_events_from_path(Path(events_path)))
    return sorted(events, key=lambda event: event.timestamp.timestamp())


def _load_anomaly_events_from_path(path: Path) -> list[AnomalyEvent]:
    try:
        raw_text = path.read_text()
    except OSError as exc:
        raise click.ClickException(f"Error loading events from {path}: {exc}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return _load_jsonl_anomaly_events(raw_text, path)
    return _coerce_anomaly_event_payload(payload, str(path))


def _load_jsonl_anomaly_events(raw_text: str, path: Path) -> list[AnomalyEvent]:
    events: list[AnomalyEvent] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Line {line_number} in {path} is not valid JSON: {exc}"
            ) from exc
        events.extend(
            _coerce_anomaly_event_payload(payload, f"{path}: line {line_number}")
        )
    return events


def _coerce_anomaly_event_payload(payload: Any, source: str) -> list[AnomalyEvent]:
    if isinstance(payload, dict):
        traces = payload.get("traces")
        if isinstance(traces, list):
            events: list[AnomalyEvent] = []
            for trace_index, trace_report in enumerate(traces):
                if not isinstance(trace_report, dict):
                    raise click.ClickException(
                        f"{source}: traces[{trace_index}] must be an object."
                    )
                trace_events = trace_report.get("events", [])
                if not isinstance(trace_events, list):
                    raise click.ClickException(
                        f"{source}: traces[{trace_index}].events must be an array."
                    )
                for event_index, event_payload in enumerate(trace_events):
                    events.append(
                        _validate_anomaly_event(
                            event_payload,
                            f"{source}: traces[{trace_index}].events[{event_index}]",
                        )
                    )
            return events
        return [_validate_anomaly_event(payload, source)]

    if isinstance(payload, list):
        return [
            _validate_anomaly_event(item, f"{source}: [{index}]")
            for index, item in enumerate(payload)
        ]

    raise click.ClickException(
        f"{source}: expected an anomaly event, event array, or analysis report."
    )


def _validate_anomaly_event(payload: Any, source: str) -> AnomalyEvent:
    if not isinstance(payload, dict):
        raise click.ClickException(f"{source}: anomaly event must be an object.")
    try:
        return AnomalyEvent.model_validate(payload)
    except Exception as exc:
        raise click.ClickException(f"{source}: invalid anomaly event: {exc}") from exc


def _build_trend_report(
    events: list[AnomalyEvent],
    source_count: int,
    window_size: int,
    escalation_threshold: float,
    fail_on_escalating: bool,
) -> dict[str, Any]:
    tracker = TrendTracker(
        window_size=window_size,
        escalation_threshold=escalation_threshold,
    )
    window_events = events[-window_size:]
    tracker.record_many(window_events)
    snapshot = tracker.snapshot()
    trend = str(snapshot["trend"])
    failed = fail_on_escalating and trend == Trend.ESCALATING.value

    return {
        "source_count": source_count,
        "event_count": len(events),
        "window_size": window_size,
        "window_event_count": len(window_events),
        "escalation_threshold": escalation_threshold,
        "trend": trend,
        "mean_severity": snapshot["mean_severity"],
        "severity_counts": _count_events_by_severity(events),
        "window_severity_counts": _count_events_by_severity(window_events),
        "first_event_at": events[0].timestamp.isoformat() if events else None,
        "last_event_at": events[-1].timestamp.isoformat() if events else None,
        "recommended_action": _recommend_trend_action(trend),
        "fail_on_escalating": fail_on_escalating,
        "failed": failed,
    }


def _recommend_trend_action(trend: str) -> str:
    actions = {
        Trend.ESCALATING.value: "investigate_current_rollout",
        Trend.STABLE.value: "continue_monitoring",
        Trend.DE_ESCALATING.value: "continue_monitoring",
        Trend.INSUFFICIENT_DATA.value: "collect_more_events",
    }
    return actions.get(trend, "investigate_current_rollout")


def _render_trend_report(report: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report, indent=2) + "\n"
    if output_format == "markdown":
        return _render_trend_markdown(report)
    return _render_trend_summary(report)


def _render_trend_summary(report: dict[str, Any]) -> str:
    lines = [
        "Anomaly trend",
        f"Sources: {report['source_count']}",
        f"Events loaded: {report['event_count']}",
        (
            f"Window: {report['window_size']} "
            f"(using {report['window_event_count']} events)"
        ),
        f"Trend: {report['trend']}",
        f"Mean severity: {_format_optional_float(report['mean_severity'])}",
        f"Recommended action: {report['recommended_action']}",
        "Severity counts: " + _format_severity_counts(report["severity_counts"]),
        "Window severity counts: "
        + _format_severity_counts(report["window_severity_counts"]),
    ]
    if report["first_event_at"]:
        lines.append(f"First event: {report['first_event_at']}")
        lines.append(f"Last event: {report['last_event_at']}")

    if report["failed"]:
        lines.append("")
        lines.append("CI gate: failed for escalating trend")
    elif report["fail_on_escalating"]:
        lines.append("")
        lines.append("CI gate: passed for escalating trend")

    return "\n".join(lines) + "\n"


def _render_trend_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Anomaly Trend",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Sources | {report['source_count']} |",
        f"| Events loaded | {report['event_count']} |",
        (
            f"| Window | {report['window_size']} "
            f"(using {report['window_event_count']} events) |"
        ),
        f"| Trend | {report['trend']} |",
        f"| Mean severity | {_format_optional_float(report['mean_severity'])} |",
        f"| Recommended action | {report['recommended_action']} |",
        f"| Severity counts | {_format_severity_counts(report['severity_counts'])} |",
        (
            f"| Window severity counts | "
            f"{_format_severity_counts(report['window_severity_counts'])} |"
        ),
    ]
    if report["first_event_at"]:
        lines.append(f"| First event | {report['first_event_at']} |")
        lines.append(f"| Last event | {report['last_event_at']} |")

    if report["fail_on_escalating"]:
        lines.extend(
            [
                "",
                "## CI Gate",
                "",
                (
                    "Failed for escalating trend."
                    if report["failed"]
                    else "Passed for escalating trend."
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _format_optional_float(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, float) else "none"


def _format_severity_counts(counts: dict[str, int]) -> str:
    return ", ".join(
        f"{severity.value}={counts[severity.value]}" for severity in Severity
    )


def _format_item_list(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _sorted_frequency_drift(
    frequency_drift: dict[str, float],
) -> list[tuple[str, float]]:
    return sorted(
        frequency_drift.items(),
        key=lambda item: (-item[1], item[0]),
    )


if __name__ == "__main__":
    main()

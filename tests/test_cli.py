"""Tests for the spectra CLI commands."""

from __future__ import annotations

import json
import random
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from spectra.cli import main
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    LLMCall,
    Severity,
    ToolCall,
    ToolStats,
)
from spectra.profiler.profile import BehavioralProfile


def _make_training_traces_json(count: int = 120) -> list[dict]:
    """Build a list of serializable trace dicts for CLI tests."""
    random.seed(42)
    traces = []
    for _ in range(count):
        started = datetime.now(UTC)
        ended = started + timedelta(minutes=2)
        trace = AgentTrace(
            agent_type="test-agent",
            started_at=started,
            ended_at=ended,
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    arguments={"query": "test"},
                    timestamp=started,
                ),
                ToolCall(
                    tool_name="respond",
                    timestamp=started + timedelta(seconds=5),
                ),
            ],
            llm_calls=[
                LLMCall(
                    model="gpt-4",
                    total_tokens=random.randint(800, 1500),
                    timestamp=started + timedelta(seconds=1),
                ),
            ],
            output="Standard response.",
        )
        traces.append(trace.model_dump(mode="json"))
    return traces


def _make_anomalous_trace_json(agent_type: str = "test-agent") -> dict:
    """Build a serializable trace that should trigger critical anomalies."""
    started = datetime.now(UTC)
    trace = AgentTrace(
        agent_type=agent_type,
        task_id="suspicious-task",
        started_at=started,
        ended_at=started + timedelta(minutes=30),
        tool_calls=[
            ToolCall(
                tool_name="database_query",
                arguments={"sql": "select * from users"},
                timestamp=started,
            ),
            ToolCall(
                tool_name="delete_record",
                arguments={"table": "users"},
                timestamp=started + timedelta(seconds=5),
            ),
        ],
        llm_calls=[
            LLMCall(
                model="gpt-4",
                total_tokens=50000,
                timestamp=started + timedelta(seconds=1),
            )
        ],
        output="```sql\nselect * from users;\n```",
    )
    return trace.model_dump(mode="json")


def _make_anomaly_event_json(severity: str, offset_seconds: int) -> dict:
    """Build a serializable anomaly event for trend CLI tests."""
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=offset_seconds)
    return AnomalyEvent(
        timestamp=timestamp,
        trace_id=f"trace-{offset_seconds}",
        agent_type="test-agent",
        detector_type=DetectorType.TOOL_USAGE,
        severity=Severity(severity),
        title=f"{severity} event",
        description="Synthetic trend fixture.",
        score=0.5,
    ).model_dump(mode="json")


def _write_profile(runner: CliRunner, tmpdir: str) -> Path:
    """Train a profile through the CLI and return its path."""
    traces_path = Path(tmpdir) / "training-traces.json"
    traces_path.write_text(json.dumps(_make_training_traces_json(120)))
    profile_path = Path(tmpdir) / "profile.json"

    result = runner.invoke(
        main,
        [
            "train",
            str(traces_path),
            "--agent-type",
            "test-agent",
            "--output",
            str(profile_path),
        ],
    )
    assert result.exit_code == 0
    return profile_path


def _write_minimal_profile(
    path: Path,
    *,
    agent_type: str = "test-agent",
    tools: dict[str, float] | None = None,
) -> Path:
    """Write a small profile fixture for drift-only CLI tests."""
    tool_means = tools or {"search_kb": 1.0, "respond": 1.0}
    profile = BehavioralProfile(
        agent_type=agent_type,
        trace_count=120,
        known_tools=set(tool_means),
        tool_stats={
            name: ToolStats(
                tool_name=name,
                usage_frequency=1.0,
                avg_calls_per_trace=avg_calls,
            )
            for name, avg_calls in tool_means.items()
        },
    )
    profile.save(path)
    return path


class TestTrainCommand:
    def test_train_success(self) -> None:
        runner = CliRunner()
        traces_data = _make_training_traces_json(120)

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "traces.json"
            traces_path.write_text(json.dumps(traces_data))
            output_path = Path(tmpdir) / "profile.json"

            result = runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                    "--output",
                    str(output_path),
                    "--min-traces",
                    "100",
                ],
            )
            assert result.exit_code == 0
            assert "Profile saved to" in result.output
            assert "Agent type: test-agent" in result.output
            assert output_path.exists()

    def test_train_with_verbose(self) -> None:
        runner = CliRunner()
        traces_data = _make_training_traces_json(120)

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "traces.json"
            traces_path.write_text(json.dumps(traces_data))
            output_path = Path(tmpdir) / "profile.json"

            result = runner.invoke(
                main,
                [
                    "-v",
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                    "--output",
                    str(output_path),
                ],
            )
            assert result.exit_code == 0

    def test_train_invalid_json(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "bad.json"
            traces_path.write_text("not valid json{{{")

            result = runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                ],
            )
            assert result.exit_code != 0

    def test_train_not_a_list(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "obj.json"
            traces_path.write_text(json.dumps({"key": "value"}))

            result = runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                ],
            )
            assert result.exit_code != 0

    def test_train_insufficient_traces(self) -> None:
        runner = CliRunner()
        traces_data = _make_training_traces_json(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "traces.json"
            traces_path.write_text(json.dumps(traces_data))

            result = runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                    "--min-traces",
                    "100",
                ],
            )
            assert result.exit_code != 0

    def test_train_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "train",
                "/nonexistent/path/traces.json",
                "--agent-type",
                "test-agent",
            ],
        )
        assert result.exit_code != 0


class TestInspectCommand:
    def test_inspect_profile(self) -> None:
        runner = CliRunner()
        random.seed(42)
        traces_data = _make_training_traces_json(120)

        with tempfile.TemporaryDirectory() as tmpdir:
            # First train a profile
            traces_path = Path(tmpdir) / "traces.json"
            traces_path.write_text(json.dumps(traces_data))
            profile_path = Path(tmpdir) / "profile.json"

            runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                    "--output",
                    str(profile_path),
                ],
            )

            # Now inspect
            result = runner.invoke(main, ["inspect", str(profile_path)])
            assert result.exit_code == 0
            assert "Agent type: test-agent" in result.output
            assert "Trace count:" in result.output
            assert "Known tools" in result.output
            assert "Volume stats:" in result.output
            assert "LLM calls:" in result.output
            assert "Tool calls:" in result.output

    def test_inspect_nonexistent_profile(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "/nonexistent/profile.json"])
        assert result.exit_code != 0


class TestAnalyzeCommand:
    def test_analyze_summary_passes_without_gate(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))

            result = runner.invoke(
                main, ["analyze", str(profile_path), str(trace_path)]
            )

            assert result.exit_code == 0
            assert "Profile: test-agent" in result.output
            assert "Anomaly events:" in result.output
            assert "[CRITICAL] Never-seen tool: database_query" in result.output

    def test_analyze_json_reports_ci_failure(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "traces.json"
            trace_path.write_text(json.dumps([_make_anomalous_trace_json()]))

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--format",
                    "json",
                    "--fail-on",
                    "HIGH",
                ],
            )

            assert result.exit_code == 2
            payload = json.loads(result.output.split("Failure threshold met:")[0])
            assert payload["failed"] is True
            assert payload["fail_on"] == "HIGH"
            assert payload["max_severity"] == "CRITICAL"
            assert payload["trace_count"] == 1
            assert payload["event_count"] > 0

    def test_analyze_jsonl_output_file_contains_events(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            output_path = Path(tmpdir) / "events.jsonl"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--format",
                    "jsonl",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert result.output == ""
            lines = output_path.read_text().splitlines()
            assert lines
            events = [json.loads(line) for line in lines]
            assert any(event["severity"] == "CRITICAL" for event in events)
            assert all(event["action_taken"] == "log" for event in events)

    def test_analyze_rejects_agent_type_mismatch_by_default(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json("other-agent")))

            result = runner.invoke(
                main, ["analyze", str(profile_path), str(trace_path)]
            )

            assert result.exit_code != 0
            assert "agent_type does not match profile" in result.output

    def test_analyze_allows_agent_type_mismatch_when_requested(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json("other-agent")))

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--allow-agent-type-mismatch",
                ],
            )

            assert result.exit_code == 0
            assert "Anomaly events:" in result.output

    def test_analyze_writes_baseline_with_stable_fingerprints(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            baseline_path = Path(tmpdir) / "baselines" / "events.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--format",
                    "json",
                    "--write-baseline",
                    str(baseline_path),
                ],
            )

            assert result.exit_code == 0
            report = json.loads(result.output)
            assert report["event_fingerprints"]
            assert report["traces"][0]["events"][0]["fingerprint"]
            baseline = json.loads(baseline_path.read_text())
            assert baseline["schema_version"] == 1
            assert baseline["event_fingerprints"] == report["event_fingerprints"]
            assert baseline["events"][0]["fingerprint"]

    def test_analyze_only_new_filters_known_baseline_events(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            baseline_path = Path(tmpdir) / "baseline.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))

            first = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--write-baseline",
                    str(baseline_path),
                ],
            )
            assert first.exit_code == 0

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--baseline",
                    str(baseline_path),
                    "--only-new",
                ],
            )

            assert result.exit_code == 0
            assert "Anomaly events: 0" in result.output
            assert "Max severity: none" in result.output
            assert "Baseline: new=0" in result.output
            assert "No anomalies detected." in result.output

    def test_analyze_fail_on_new_ignores_unchanged_baseline_events(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            baseline_path = Path(tmpdir) / "baseline.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))

            first = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--write-baseline",
                    str(baseline_path),
                ],
            )
            assert first.exit_code == 0

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--baseline",
                    str(baseline_path),
                    "--fail-on",
                    "LOW",
                    "--fail-on-new",
                ],
            )

            assert result.exit_code == 0
            assert "Baseline: new=0" in result.output
            assert "CI gate: passed for new events at LOW" in result.output

    def test_analyze_fail_on_new_fails_for_new_baseline_events(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            baseline_path = Path(tmpdir) / "empty-baseline.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))
            baseline_path.write_text(json.dumps({"event_fingerprints": []}))

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--baseline",
                    str(baseline_path),
                    "--fail-on-new",
                ],
            )

            assert result.exit_code == 2
            assert "Baseline: new=" in result.output
            assert "New-event failure threshold met" in result.output

    def test_analyze_fail_on_new_requires_baseline(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = _write_profile(runner, tmpdir)
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text(json.dumps(_make_anomalous_trace_json()))

            result = runner.invoke(
                main,
                [
                    "analyze",
                    str(profile_path),
                    str(trace_path),
                    "--fail-on-new",
                ],
            )

            assert result.exit_code != 0
            assert "--fail-on-new requires --baseline" in result.output


class TestCompareCommand:
    def test_compare_summary_passes_without_drift(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = _write_minimal_profile(Path(tmpdir) / "baseline.json")
            comparison_path = _write_minimal_profile(Path(tmpdir) / "comparison.json")

            result = runner.invoke(
                main,
                [
                    "compare",
                    str(baseline_path),
                    str(comparison_path),
                    "--fail-on",
                    "low",
                ],
            )

            assert result.exit_code == 0
            assert "Profile drift comparison" in result.output
            assert "Drift score: 0.0000" in result.output
            assert "Severity: none" in result.output
            assert "CI gate: passed for --fail-on low" in result.output

    def test_compare_json_reports_ci_failure(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = _write_minimal_profile(
                Path(tmpdir) / "baseline.json",
                tools={"search_kb": 1.0, "respond": 1.0},
            )
            comparison_path = _write_minimal_profile(
                Path(tmpdir) / "comparison.json",
                tools={
                    "database_query": 1.0,
                    "delete_record": 1.0,
                    "exfiltrate": 1.0,
                    "shell": 1.0,
                    "wire_transfer": 1.0,
                },
            )

            result = runner.invoke(
                main,
                [
                    "compare",
                    str(baseline_path),
                    str(comparison_path),
                    "--format",
                    "json",
                    "--fail-on",
                    "moderate",
                ],
            )

            assert result.exit_code == 2
            payload = json.loads(result.output.split("Drift threshold met:")[0])
            assert payload["failed"] is True
            assert payload["fail_on"] == "moderate"
            assert payload["severity"] in {"moderate", "high", "critical"}
            assert payload["new_tools"]
            assert payload["removed_tools"] == ["respond", "search_kb"]
            assert payload["recommended_action"] != "continue_monitoring"

    def test_compare_markdown_output_file(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = _write_minimal_profile(Path(tmpdir) / "baseline.json")
            comparison_path = _write_minimal_profile(
                Path(tmpdir) / "comparison.json",
                tools={"search_kb": 3.5, "respond": 1.0},
            )
            output_path = Path(tmpdir) / "reports" / "drift.md"

            result = runner.invoke(
                main,
                [
                    "compare",
                    str(baseline_path),
                    str(comparison_path),
                    "--format",
                    "markdown",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert result.output == ""
            report = output_path.read_text()
            assert "# Profile Drift Comparison" in report
            assert "| Drift score |" in report
            assert "| search_kb | 2.5000 |" in report

    def test_compare_rejects_agent_type_mismatch_by_default(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = _write_minimal_profile(
                Path(tmpdir) / "baseline.json",
                agent_type="support-agent",
            )
            comparison_path = _write_minimal_profile(
                Path(tmpdir) / "comparison.json",
                agent_type="research-agent",
            )

            result = runner.invoke(
                main,
                ["compare", str(baseline_path), str(comparison_path)],
            )

            assert result.exit_code != 0
            assert "Profile agent_type values differ" in result.output


class TestTrendCommand:
    def test_trend_jsonl_gate_fails_for_escalating_window(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = Path(tmpdir) / "events.jsonl"
            events = [
                _make_anomaly_event_json("LOW", 1),
                _make_anomaly_event_json("LOW", 2),
                _make_anomaly_event_json("HIGH", 3),
                _make_anomaly_event_json("CRITICAL", 4),
            ]
            events_path.write_text("\n".join(json.dumps(event) for event in events))

            result = runner.invoke(
                main,
                [
                    "trend",
                    str(events_path),
                    "--window-size",
                    "4",
                    "--format",
                    "json",
                    "--fail-on-escalating",
                ],
            )

            assert result.exit_code == 2
            payload = json.loads(result.output.split("Trend gate failed:")[0])
            assert payload["trend"] == "escalating"
            assert payload["failed"] is True
            assert payload["window_event_count"] == 4
            assert payload["window_severity_counts"]["CRITICAL"] == 1

    def test_trend_summary_reads_analysis_report_events(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "analysis.json"
            report_path.write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "trace_id": "trace-a",
                                "events": [
                                    _make_anomaly_event_json("LOW", 1),
                                    _make_anomaly_event_json("LOW", 2),
                                ],
                            },
                            {
                                "trace_id": "trace-b",
                                "events": [
                                    _make_anomaly_event_json("LOW", 3),
                                    _make_anomaly_event_json("LOW", 4),
                                ],
                            },
                        ]
                    }
                )
            )

            result = runner.invoke(
                main,
                [
                    "trend",
                    str(report_path),
                    "--window-size",
                    "4",
                    "--fail-on-escalating",
                ],
            )

            assert result.exit_code == 0
            assert "Anomaly trend" in result.output
            assert "Events loaded: 4" in result.output
            assert "Trend: stable" in result.output
            assert "CI gate: passed for escalating trend" in result.output

    def test_trend_markdown_output_file(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = Path(tmpdir) / "events.json"
            output_path = Path(tmpdir) / "reports" / "trend.md"
            events_path.write_text(
                json.dumps(
                    [
                        _make_anomaly_event_json("MEDIUM", 1),
                        _make_anomaly_event_json("MEDIUM", 2),
                        _make_anomaly_event_json("LOW", 3),
                        _make_anomaly_event_json("LOW", 4),
                    ]
                )
            )

            result = runner.invoke(
                main,
                [
                    "trend",
                    str(events_path),
                    "--window-size",
                    "4",
                    "--format",
                    "markdown",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert result.output == ""
            report = output_path.read_text()
            assert "# Anomaly Trend" in report
            assert "| Trend | de-escalating |" in report
            assert (
                "| Window severity counts | LOW=2, MEDIUM=2, HIGH=0, CRITICAL=0 |"
                in report
            )


class TestDashboardCommand:
    def test_dashboard_nonexistent_profile(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["dashboard", "/nonexistent/profile.json"])
        assert result.exit_code != 0

    def test_dashboard_launches(self) -> None:
        runner = CliRunner()
        random.seed(42)
        traces_data = _make_training_traces_json(120)

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "traces.json"
            traces_path.write_text(json.dumps(traces_data))
            profile_path = Path(tmpdir) / "profile.json"

            runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                    "--output",
                    str(profile_path),
                ],
            )

            with patch("uvicorn.run") as mock_run:
                result = runner.invoke(
                    main,
                    [
                        "dashboard",
                        str(profile_path),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9000",
                    ],
                )
                assert result.exit_code == 0
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["host"] == "0.0.0.0"
                assert call_kwargs[1]["port"] == 9000

    def test_dashboard_no_uvicorn(self) -> None:
        runner = CliRunner()
        random.seed(42)
        traces_data = _make_training_traces_json(120)

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path = Path(tmpdir) / "traces.json"
            traces_path.write_text(json.dumps(traces_data))
            profile_path = Path(tmpdir) / "profile.json"

            runner.invoke(
                main,
                [
                    "train",
                    str(traces_path),
                    "--agent-type",
                    "test-agent",
                    "--output",
                    str(profile_path),
                ],
            )

            with patch.dict("sys.modules", {"uvicorn": None}):
                result = runner.invoke(
                    main,
                    ["dashboard", str(profile_path)],
                )
                assert result.exit_code != 0


class TestVersionOption:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

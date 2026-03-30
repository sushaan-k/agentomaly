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
from spectra.models import AgentTrace, LLMCall, ToolCall


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

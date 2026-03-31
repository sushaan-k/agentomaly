"""Tests for JSON Lines export of anomaly events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectra.models import AgentTrace
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile


class TestJSONLExport:
    @pytest.mark.asyncio
    async def test_to_jsonl_creates_file(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """to_jsonl should write a JSONL file with one line per event."""
        monitor = Monitor(
            profile=trained_profile,
            response_policy={
                "LOW": "log",
                "MEDIUM": "log",
                "HIGH": "log",
                "CRITICAL": "log",
            },
        )
        monitor.start()
        await monitor.analyze(anomalous_trace)

        out = tmp_path / "events.jsonl"
        count = monitor.to_jsonl(out)

        assert out.exists()
        assert count > 0

        lines = out.read_text().strip().splitlines()
        assert len(lines) == count

        # Every line should be valid JSON with expected fields
        for line in lines:
            obj = json.loads(line)
            assert "event_id" in obj
            assert "severity" in obj
            assert "trace_id" in obj
            assert "detector_type" in obj

    @pytest.mark.asyncio
    async def test_to_jsonl_empty_log(
        self,
        trained_profile: BehavioralProfile,
        tmp_path: Path,
    ) -> None:
        """to_jsonl with no events should create an empty file."""
        monitor = Monitor(profile=trained_profile)
        monitor.start()

        out = tmp_path / "empty.jsonl"
        count = monitor.to_jsonl(out)

        assert count == 0
        assert out.exists()
        assert out.read_text() == ""

    @pytest.mark.asyncio
    async def test_to_jsonl_creates_parent_dirs(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """to_jsonl should create parent directories automatically."""
        monitor = Monitor(
            profile=trained_profile,
            response_policy={
                "LOW": "log",
                "MEDIUM": "log",
                "HIGH": "log",
                "CRITICAL": "log",
            },
        )
        monitor.start()
        await monitor.analyze(anomalous_trace)

        nested = tmp_path / "deep" / "nested" / "dir" / "events.jsonl"
        count = monitor.to_jsonl(nested)
        assert count > 0
        assert nested.exists()

    @pytest.mark.asyncio
    async def test_to_jsonl_events_are_deserializable(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """Each JSONL record should be loadable back into an AnomalyEvent."""
        from spectra.models import AnomalyEvent

        monitor = Monitor(
            profile=trained_profile,
            response_policy={
                "LOW": "log",
                "MEDIUM": "log",
                "HIGH": "log",
                "CRITICAL": "log",
            },
        )
        monitor.start()
        await monitor.analyze(anomalous_trace)

        out = tmp_path / "roundtrip.jsonl"
        monitor.to_jsonl(out)

        for line in out.read_text().strip().splitlines():
            data = json.loads(line)
            event = AnomalyEvent.model_validate(data)
            assert event.event_id == data["event_id"]

    @pytest.mark.asyncio
    async def test_to_jsonl_multiple_analyses(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """Events from multiple analyze() calls should all be exported."""
        monitor = Monitor(
            profile=trained_profile,
            response_policy={
                "LOW": "log",
                "MEDIUM": "log",
                "HIGH": "log",
                "CRITICAL": "log",
            },
        )
        monitor.start()
        await monitor.analyze(anomalous_trace)
        await monitor.analyze(anomalous_trace)

        out = tmp_path / "multi.jsonl"
        count = monitor.to_jsonl(out)

        assert count == len(monitor.event_log)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == count

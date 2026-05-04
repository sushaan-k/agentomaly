"""Stable anomaly fingerprints and baseline comparison helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spectra.models import AnomalyEvent

_FINGERPRINT_FIELDS = (
    "trace_id",
    "agent_type",
    "detector_type",
    "severity",
    "title",
    "description",
    "details",
)


def event_fingerprint(event: AnomalyEvent | Mapping[str, Any]) -> str:
    """Return a stable fingerprint for an anomaly event.

    Runtime-generated fields such as ``event_id``, ``timestamp``,
    ``score``, and ``action_taken`` are intentionally excluded so the same
    detector finding keeps the same identity across repeated offline runs.
    """
    if isinstance(event, AnomalyEvent):
        event_data = event.model_dump(mode="json")
    else:
        event_data = dict(event)

    payload = {
        field: _normalize(event_data.get(field))
        for field in _FINGERPRINT_FIELDS
        if field in event_data
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def annotate_event(event: dict[str, Any]) -> str:
    """Add and return the stable fingerprint for a serialized event."""
    fingerprint = event_fingerprint(event)
    event["fingerprint"] = fingerprint
    return fingerprint


def annotate_report(report: dict[str, Any]) -> list[str]:
    """Add fingerprints to every event in a CLI analysis report."""
    fingerprints: list[str] = []
    for event in iter_report_events(report):
        fingerprints.append(annotate_event(event))
    report["event_fingerprints"] = sorted(fingerprints)
    return fingerprints


def iter_report_events(report: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield serialized event dictionaries from a CLI analysis report."""
    traces = report.get("traces", [])
    if not isinstance(traces, list):
        return
    for trace_report in traces:
        if not isinstance(trace_report, dict):
            continue
        events = trace_report.get("events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict):
                yield event


def load_baseline_fingerprints(path: str | Path) -> set[str]:
    """Load event fingerprints from a baseline or analysis JSON file."""
    baseline_path = Path(path)
    try:
        raw = json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not load baseline {baseline_path}: {exc}") from exc

    return _extract_fingerprints(raw)


def baseline_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact baseline file from an annotated analysis report."""
    fingerprints = sorted(
        str(fingerprint) for fingerprint in report.get("event_fingerprints", [])
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_agent_type": report.get("profile_agent_type"),
        "sensitivity": report.get("sensitivity"),
        "trace_count": report.get("trace_count"),
        "event_count": len(fingerprints),
        "event_fingerprints": fingerprints,
        "events": [
            {
                "fingerprint": event.get("fingerprint"),
                "trace_id": event.get("trace_id"),
                "agent_type": event.get("agent_type"),
                "detector_type": event.get("detector_type"),
                "severity": event.get("severity"),
                "title": event.get("title"),
            }
            for event in iter_report_events(report)
            if event.get("fingerprint") in fingerprints
        ],
    }


def write_baseline(report: Mapping[str, Any], path: str | Path) -> None:
    """Write a compact anomaly baseline JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline_payload(report), indent=2) + "\n")


def compare_report_to_baseline(
    report: dict[str, Any],
    baseline_fingerprints: set[str],
) -> dict[str, Any]:
    """Annotate report events with baseline status and return comparison data."""
    annotate_report(report)
    current_fingerprints = set(report["event_fingerprints"])

    for event in iter_report_events(report):
        fingerprint = str(event["fingerprint"])
        event["baseline_status"] = (
            "unchanged" if fingerprint in baseline_fingerprints else "new"
        )

    new_fingerprints = sorted(current_fingerprints - baseline_fingerprints)
    unchanged_fingerprints = sorted(current_fingerprints & baseline_fingerprints)
    resolved_fingerprints = sorted(baseline_fingerprints - current_fingerprints)
    comparison = {
        "baseline_event_count": len(baseline_fingerprints),
        "current_event_count": len(current_fingerprints),
        "new_event_count": len(new_fingerprints),
        "unchanged_event_count": len(unchanged_fingerprints),
        "resolved_event_count": len(resolved_fingerprints),
        "new_fingerprints": new_fingerprints,
        "unchanged_fingerprints": unchanged_fingerprints,
        "resolved_fingerprints": resolved_fingerprints,
    }
    report["baseline_comparison"] = comparison
    return comparison


def _extract_fingerprints(raw: Any) -> set[str]:
    if isinstance(raw, dict):
        direct = raw.get("event_fingerprints")
        if isinstance(direct, list):
            return {str(item) for item in direct}

        fingerprints: set[str] = set()
        for event in iter_report_events(raw):
            if isinstance(event.get("fingerprint"), str):
                fingerprints.add(str(event["fingerprint"]))
            else:
                fingerprints.add(event_fingerprint(event))
        if fingerprints:
            return fingerprints

        events = raw.get("events")
        if isinstance(events, list):
            return _extract_event_list(events)

    if isinstance(raw, list):
        return _extract_event_list(raw)

    raise ValueError("Baseline must contain event_fingerprints or event objects.")


def _extract_event_list(events: list[Any]) -> set[str]:
    fingerprints: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("fingerprint"), str):
            fingerprints.add(str(event["fingerprint"]))
        else:
            fingerprints.add(event_fingerprint(event))
    return fingerprints


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize(child) for child in value]
    if isinstance(value, tuple):
        return [_normalize(child) for child in value]
    return value

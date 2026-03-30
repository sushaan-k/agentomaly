# Tuning

`spectra` is intentionally opinionated out of the box, but the runtime is tunable in a few places that matter for production use.

## Start with profile quality

The monitor is only as good as the baseline profile. Train on representative traces from normal production behavior:

- the right agent type
- the right task mix
- enough historical variety to cover legitimate tool usage

If the baseline is too narrow, the detectors will over-flag normal work. If it is too broad, they will miss real drift.

## Use the right sensitivity

`Monitor` accepts `sensitivity="low" | "medium" | "high" | "paranoid"`.

- `low` is the least aggressive
- `medium` is the default
- `high` increases sensitivity without going fully noisy
- `paranoid` is for security-heavy workflows where false positives are acceptable

## Route responses by severity

`response_policy` controls what happens after a detector emits an anomaly:

```python
from spectra import Monitor

monitor = Monitor(
    profile=profile,
    response_policy={
        "LOW": "log",
        "MEDIUM": "alert",
        "HIGH": "quarantine",
        "CRITICAL": "block",
    },
)
```

The policy accepts either enum keys and values or string shorthands.

## Tune detector-specific behavior

If you need more precision, customize the detectors directly:

- `ToolAnomalyDetector(thresholds=...)`
- `SequenceAnomalyDetector(loop_threshold=...)`
- `VolumeAnomalyDetector(thresholds=...)`
- `ContentAnomalyDetector(structure_threshold=...)`
- `InjectionDetector(shift_window=...)`

You can pass a custom detector list to `Monitor(detectors=[...])` to replace the default five-detector bundle.

## Separate alerting from blocking

Alert channels and blocking are independent. Use `alert_channels` for notifications and `TaskBlocker` for runtime interruption or quarantine:

```python
from spectra.response.blocker import TaskBlocker

async def on_block(event):
    ...

async def on_quarantine(event):
    ...

blocker = TaskBlocker(
    on_block=on_block,
    on_quarantine=on_quarantine,
)
```

If you want a conservative first deployment, start with `LogChannel` and a quarantine-oriented response policy before enabling hard blocks.

## Watch the event log

`Monitor.event_log` returns the full in-memory anomaly history, and `Monitor.summary()` gives you counts by severity. Use those outputs to tighten the baseline and detector knobs over time.

# Detectors

`spectra` ships with five built-in detectors. Each detector implements the shared `BaseDetector.analyze(trace, profile)` interface and returns zero or more `AnomalyEvent` objects.

## Built-in detectors

### `ToolAnomalyDetector`

Flags:

- never-seen tools
- frequency spikes or drops compared to the learned baseline
- unusual argument keys for known tools

Constructor knobs:

- `sensitivity`
- `thresholds=SensitivityThresholds()`

### `SequenceAnomalyDetector`

Flags:

- novel action transitions
- low-probability action sequences
- repeated loops

Constructor knobs:

- `sensitivity`
- `thresholds=SensitivityThresholds()`
- `loop_threshold=3`

### `VolumeAnomalyDetector`

Flags:

- LLM call count anomalies
- tool call count anomalies
- total token usage anomalies
- wall-clock duration anomalies

Constructor knobs:

- `sensitivity`
- `thresholds=SensitivityThresholds()`

### `ContentAnomalyDetector`

Flags:

- unexpected code blocks
- unexpected URLs
- unexpected structured data
- large output-length deviations

Constructor knobs:

- `sensitivity`
- `thresholds=SensitivityThresholds()`
- `structure_threshold=0.05`

### `InjectionDetector`

Looks for behavioral shifts that often follow prompt injection or tainted tool output.

Constructor knobs:

- `sensitivity`
- `shift_window=3`

## Sensitivity levels

`Sensitivity` is a string enum with these values:

- `low`
- `medium`
- `high`
- `paranoid`

The detectors use `SensitivityThresholds` to translate those presets into z-score cutoffs.

## Custom detector sets

Pass your own detector list to `Monitor` if you want to disable or replace the defaults:

```python
from spectra import Monitor
from spectra.detectors import ToolAnomalyDetector, VolumeAnomalyDetector

monitor = Monitor(
    profile=profile,
    detectors=[
        ToolAnomalyDetector(sensitivity="high"),
        VolumeAnomalyDetector(sensitivity="high"),
    ],
)
```

If `detectors` is omitted, `Monitor` enables the five built-in detectors.

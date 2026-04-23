# agentomaly

[![CI](https://github.com/sushaan-k/agentomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/agentomaly/actions)
[![PyPI](https://img.shields.io/pypi/v/agentomaly.svg)](https://pypi.org/project/agentomaly/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/agentomaly.svg)](https://pypi.org/project/agentomaly/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-compatible-blueviolet.svg)](https://opentelemetry.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Runtime behavioral anomaly detection for AI agents — via OpenTelemetry.**

`agentomaly` wraps your agent runs with OpenTelemetry instrumentation, builds behavioral baselines from historical traces, and fires alerts when an agent deviates: unexpected tool call sequences, cost explosions, latency regressions, or output distribution shifts.

---

## The Problem

Production AI agents fail silently. A model starts calling tools in unusual sequences, costs double per run, or its output distribution shifts after a model provider update — and you find out from a user complaint, not a metric. APM tools (Datadog, New Relic) understand HTTP latency. They do not understand "this agent normally calls `search` before `write` but today it's calling `write` before `search` 40% of the time." Nobody has built the behavioral baseline layer for agents.

## Solution

```python
from spectra import AgentObserver, BaselineTracker, AnomalyDetector

# Wrap any agent function with OpenTelemetry instrumentation
observer = AgentObserver(export_to="jaeger://localhost:4317")

@observer.trace
async def my_agent(task: str) -> str:
    # your existing agent code — unchanged
    ...

# After collecting a baseline (50+ runs), enable anomaly detection
detector = AnomalyDetector(
    baseline=BaselineTracker.load("my_agent_baseline.json"),
    thresholds={
        "tool_call_sequence_divergence": 0.3,   # Jensen-Shannon divergence
        "cost_zscore": 3.0,                      # 3σ cost spike
        "latency_regression_p99_pct": 0.5,       # 50% P99 regression
    },
)

detector.attach(observer)
detector.on_anomaly(lambda a: print(f"[ALERT] {a.kind}: {a.description}"))
# [ALERT] cost_spike: run cost $0.84 vs baseline $0.12 (7.0σ)
# [ALERT] sequence_divergence: tool_call sequence JS divergence 0.41 > threshold 0.30
```

## At a Glance

- **OTel-native** — emits spans and metrics to any OpenTelemetry backend
- **Behavioral baselines** — learns normal tool call sequences, costs, latencies, and output lengths
- **Statistical detection** — Jensen-Shannon divergence for sequences, z-score for scalar metrics
- **Zero-code instrumentation** — wrap existing agent functions without modifying internals
- **Alert routing** — webhooks, PagerDuty, Slack, or custom callbacks

## Install

```bash
pip install agentomaly
```

## Detectors

| Detector | Metric | Method |
|---|---|---|
| Tool Sequence | Call order distribution | Jensen-Shannon divergence |
| Cost Spike | Token spend per run | Z-score (rolling window) |
| Latency Regression | P50/P95/P99 step latency | Welch's t-test |
| Output Distribution | Response length + vocabulary | KL divergence |
| Retry Storm | Retry rate per tool | Control chart |

## Architecture

```
AgentObserver
 ├── Instrumentor      # wraps agent callables with OTel spans
 ├── SpanProcessor     # extracts behavioral features from trace
 ├── BaselineTracker   # stores and updates rolling behavioral baseline
 ├── AnomalyDetector   # statistical tests against baseline
 └── AlertRouter       # webhook / Slack / PagerDuty dispatch
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐

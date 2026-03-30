# Getting Started

`spectra` is a runtime behavioral anomaly detector for AI agents. It learns a baseline from historical traces, monitors new traces, and emits `AnomalyEvent` objects that can be logged, sent to webhooks, or routed into blocking/quarantine workflows.

## Install

```bash
pip install spectra-ai
```

For the dashboard:

```bash
pip install "spectra-ai[dashboard]"
```

## Train a profile

Train from a list of serialized `AgentTrace` objects:

```python
from spectra.models import AgentTrace
from spectra.profiler.trainer import ProfileTrainer

traces = [AgentTrace.model_validate(item) for item in historical_traces]
trainer = ProfileTrainer(min_traces=100)
profile = trainer.train(agent_type="customer-support", traces=traces)
profile.save("customer_support_profile.json")
```

## Monitor traces

```python
from spectra import Monitor, Profile

profile = Profile.load("customer_support_profile.json")
monitor = Monitor(
    profile=profile,
    sensitivity="medium",
    response_policy={
        "LOW": "log",
        "MEDIUM": "alert",
        "HIGH": "quarantine",
        "CRITICAL": "block",
    },
)
monitor.start()

events = await monitor.analyze(agent_trace)
for event in events:
    print(event.severity.value, event.title, event.score)
```

`Monitor.start()` is required before `analyze()`. If you want a lightweight local setup, omit `alert_channels` and the monitor defaults to `LogChannel()`.

## Instrument an agent

Use the `@spectra.trace` decorator and record tool or LLM calls inside the traced function:

```python
import spectra


@spectra.trace(agent_type="customer-support")
async def handle_request(user_message: str) -> str:
    spectra.record_tool_call(
        tool_name="search_kb",
        arguments={"query": user_message},
    )
    spectra.record_llm_call(model="gpt-4", total_tokens=1500)
    return "Response"
```

## CLI

```bash
spectra train traces.json --agent-type customer-support --output profile.json
spectra inspect profile.json
spectra dashboard profile.json --port 8400
```

The dashboard command requires `uvicorn` and `fastapi` via the `dashboard` extra.

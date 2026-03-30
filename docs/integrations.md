# Integrations

`spectra` is designed to fit into existing observability and agent tooling. The package currently exposes instrumentation helpers, a dashboard, and multiple alert channels.

## OpenTelemetry

Use `OTelCollector` to export traces and anomalies as spans:

```python
from spectra.instrumentation import OTelCollector

collector = OTelCollector(service_name="my-agent")
collector.export_trace(agent_trace)
collector.export_anomaly(anomaly_event)
```

The collector uses the active OpenTelemetry tracer provider when one is not supplied explicitly.

## LangGraph

Use `LangGraphCallback` to record node activity, tool calls, and LLM calls into the active `@spectra.trace` context:

```python
from spectra.instrumentation.langgraph import LangGraphCallback

callback = LangGraphCallback(agent_type="research-agent")
callback.on_tool_start("search", {"query": "test"})
callback.on_tool_end("search", result="Found results")
```

## MCP

Use `MCPMiddleware` to wrap async MCP tool handlers:

```python
from spectra.instrumentation.mcp import MCPMiddleware

middleware = MCPMiddleware(agent_type="assistant")

@middleware.wrap_tool("file_search")
async def file_search(query: str) -> str:
    ...
```

## Alert channels

Alert delivery is handled by `AlertChannel` implementations. The response policy will call each configured channel for `ALERT`, `QUARANTINE`, and `BLOCK` actions.

Available channels:

- `LogChannel`
- `WebhookChannel`
- `SlackWebhook`
- `PagerDutyChannel`

Example:

```python
from spectra import Monitor, PagerDutyChannel, SlackWebhook, WebhookChannel

monitor = Monitor(
    profile=profile,
    alert_channels=[
        PagerDutyChannel(routing_key="PD_ROUTING_KEY"),
        SlackWebhook(webhook_url="https://hooks.slack.com/services/..."),
        WebhookChannel(url="https://your-api.com/alerts"),
    ],
)
```

`PagerDutyChannel` sends PagerDuty Events API v2 trigger events. It requires a routing key, and defaults to `source="spectra"` and the standard PagerDuty enqueue endpoint.

## Dashboard

The dashboard lives in `spectra.dashboard.app` and is exposed through the CLI:

```bash
spectra dashboard profile.json --port 8400
```

Install `spectra-ai[dashboard]` to get `fastapi` and `uvicorn`.

# spectra

## Runtime Behavioral Anomaly Detection for AI Agents

### The Problem

AI agents are being deployed into production at an accelerating rate — Gartner predicts 40% of enterprise applications will embed AI agents by end of 2026. But there's a critical blind spot: **nobody is monitoring what these agents actually do**.

Traditional application monitoring (Datadog, New Relic, PagerDuty) tracks infrastructure metrics: CPU, memory, latency, error rates. But an agent can be running perfectly fine on infrastructure metrics while doing something catastrophically wrong:
- Making unusual tool calls it's never made before
- Accessing data it shouldn't need for this task
- Taking 50 actions when similar tasks usually take 10
- Producing outputs that are structurally different from past outputs
- Being manipulated by indirect prompt injection (the action looks "normal" to infra monitoring)

Torq's Socrates platform showed that AI can automate 90% of Tier-1 security analyst tasks. But who watches the watchers? When an agent starts behaving anomalously — because it's been compromised, because the model was updated, because the system prompt was changed, because user input triggered unexpected behavior — there's no alarm.

### The Solution

`spectra` is a lightweight observability layer that attaches to any AI agent runtime, learns normal behavioral patterns, and flags anomalies in real-time. It's "security monitoring for the agent era."

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        spectra                            │
│                                                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Instrumentation Layer                 │     │
│  │                                                   │     │
│  │  Lightweight middleware that captures:              │     │
│  │  - Every LLM call (prompt, response, model, tokens)│     │
│  │  - Every tool call (name, args, result, duration)  │     │
│  │  - Every state transition (what changed)           │     │
│  │  - Task metadata (type, user, session)             │     │
│  │                                                   │     │
│  │  Integration points:                               │     │
│  │  - Python decorator (@spectra.trace)               │     │
│  │  - OpenTelemetry collector                         │     │
│  │  - MCP middleware                                  │     │
│  │  - LangGraph callback                              │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Behavioral Profiler                   │     │
│  │                                                   │     │
│  │  Learns "normal" agent behavior from historical    │     │
│  │  traces. Builds a behavioral profile:              │     │
│  │                                                   │     │
│  │  ┌────────────────────────────────────────────┐   │     │
│  │  │ Profile: "customer-support-agent"           │   │     │
│  │  │                                            │   │     │
│  │  │ Tool usage:                                │   │     │
│  │  │   search_kb: 95% of tasks, avg 2.3 calls  │   │     │
│  │  │   create_ticket: 40% of tasks, avg 1.0     │   │     │
│  │  │   send_email: 15% of tasks, avg 1.0        │   │     │
│  │  │   database_query: 0% of tasks ← NEVER USED │   │     │
│  │  │                                            │   │     │
│  │  │ Action sequences:                          │   │     │
│  │  │   search → respond: 60%                    │   │     │
│  │  │   search → ticket → respond: 25%           │   │     │
│  │  │   search → email → respond: 10%            │   │     │
│  │  │                                            │   │     │
│  │  │ Step count: μ=4.2, σ=1.8, max=12           │   │     │
│  │  │ Token usage: μ=2100, σ=800                 │   │     │
│  │  │ Response time: μ=3.2s, σ=1.1s              │   │     │
│  │  └────────────────────────────────────────────┘   │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │            Anomaly Detection Engine                │     │
│  │                                                   │     │
│  │  Real-time comparison of current behavior          │     │
│  │  against the learned profile:                      │     │
│  │                                                   │     │
│  │  Detectors:                                        │     │
│  │  ┌──────────────────────────────────────────┐     │     │
│  │  │ Tool Anomaly: Agent called database_query │     │     │
│  │  │   → This tool has NEVER been used before  │     │     │
│  │  │   → Severity: CRITICAL                    │     │     │
│  │  └──────────────────────────────────────────┘     │     │
│  │  ┌──────────────────────────────────────────┐     │     │
│  │  │ Sequence Anomaly: search → database_query │     │     │
│  │  │   → delete_record → respond               │     │     │
│  │  │   → Never-seen action sequence            │     │     │
│  │  │   → Severity: CRITICAL                    │     │     │
│  │  └──────────────────────────────────────────┘     │     │
│  │  ┌──────────────────────────────────────────┐     │     │
│  │  │ Volume Anomaly: 23 tool calls this task   │     │     │
│  │  │   → Normal: 4.2 ± 1.8 (>5σ deviation)    │     │     │
│  │  │   → Severity: HIGH                        │     │     │
│  │  └──────────────────────────────────────────┘     │     │
│  │  ┌──────────────────────────────────────────┐     │     │
│  │  │ Content Anomaly: Response contains code   │     │     │
│  │  │   block — agent has never output code      │     │     │
│  │  │   → Severity: MEDIUM                      │     │     │
│  │  └──────────────────────────────────────────┘     │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Response Engine                       │     │
│  │                                                   │     │
│  │  - Alert (webhook, Slack, PagerDuty, email)        │     │
│  │  - Block (kill the agent task immediately)         │     │
│  │  - Quarantine (pause and flag for human review)    │     │
│  │  - Log (record for later analysis)                 │     │
│  │  - Explain (generate human-readable explanation    │     │
│  │    of why this behavior is anomalous)              │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Anomaly Detectors (Detail)

#### 1. Tool Usage Anomaly
Tracks which tools each agent type uses and flags:
- **Never-seen tools**: Agent uses a tool it has never used before (strongest signal)
- **Frequency anomaly**: Agent uses a tool far more/less than normal
- **Argument anomaly**: Tool arguments deviate from normal patterns (e.g., usually queries user's own data, now querying another user's data)

#### 2. Action Sequence Anomaly
Builds a Markov model of typical action sequences and flags:
- **Novel sequences**: Action chain never seen in training data
- **Abnormal transitions**: Transition probability < threshold (e.g., `delete` after `search` when normally it's `respond` after `search`)
- **Loop detection**: Agent stuck in a tool-call loop

#### 3. Volume / Duration Anomaly
Statistical anomaly detection on:
- Number of LLM calls per task
- Number of tool calls per task
- Total tokens consumed
- Wall-clock time
- Cost

Uses z-score with adaptive thresholds (configurable sensitivity).

#### 4. Content Anomaly
Analyzes the semantic content of agent outputs:
- **Output structure change**: Response format deviates from normal (e.g., suddenly contains code, URLs, or structured data when it normally outputs prose)
- **Topic drift**: Agent's response is about something entirely different from the task
- **Sensitivity escalation**: Output contains information that's more sensitive than usual for this agent type

#### 5. Prompt Injection Detection
Specific detector for indirect prompt injection attacks:
- Monitors for sudden behavioral shifts mid-conversation
- Flags when agent behavior changes after processing external content (tool results, web pages, documents)
- Correlates behavioral anomalies with external content ingestion timestamps

### Technical Stack

- **Language**: Python 3.11+
- **Anomaly detection**: Z-score statistical analysis, custom Markov chain behavioral models
- **Tracing**: OpenTelemetry (compatible with existing observability stacks)
- **Alerting**: Webhooks, Slack integration (with rate limiting)
- **Dashboard**: `fastapi` + lightweight web UI (with optional API key auth)

### API Surface (Draft)

```python
from spectra import Monitor, Profile

# Attach to an agent (decorator style)
@spectra.trace(agent_type="customer-support")
async def handle_request(user_message: str):
    # ... your agent code ...
    pass

# Or attach as middleware
from spectra.integrations import LangGraphMiddleware
graph = StateGraph(...)
graph.add_middleware(LangGraphMiddleware(agent_type="customer-support"))

# Or attach to MCP
from spectra.integrations import MCPMiddleware
mcp_server.add_middleware(MCPMiddleware(agent_type="customer-support"))

# Train a behavioral profile
profile = Profile.train(
    agent_type="customer-support",
    traces=historical_traces,           # List of past execution traces
    min_traces=100,                     # Need at least 100 traces
)

# Configure monitoring
monitor = Monitor(
    profile=profile,
    sensitivity="medium",               # low / medium / high / paranoid
    response_policy={
        "CRITICAL": "block",            # Kill the task
        "HIGH": "quarantine",           # Pause for human review
        "MEDIUM": "alert",              # Send alert, let it continue
        "LOW": "log",                   # Just log it
    },
    alert_channels=[
        spectra.SlackWebhook("https://hooks.slack.com/..."),
        spectra.PagerDuty(api_key="..."),
    ],
)

# Start monitoring
monitor.start()
```

### Alert Example

```
🚨 CRITICAL ANOMALY — customer-support-agent

Task: #4521 (user: alice@company.com)
Time: 2026-03-29T14:23:15Z

Detection: Tool Usage Anomaly
  Agent called `database_query` with args:
    query: "SELECT * FROM users WHERE role='admin'"

  This tool has NEVER been used by this agent type.
  This query pattern requests sensitive data (admin users).

Context:
  The anomaly occurred immediately after the agent processed
  a customer email containing the text: "Please run this
  database check for maintenance purposes..."

  ⚠️ POSSIBLE INDIRECT PROMPT INJECTION

Action Taken: BLOCKED (task terminated)
Trace ID: abc-123-def-456

[View Full Trace] [Mark False Positive] [Investigate]
```

### What Makes This Novel

1. **Entirely new category** — "behavioral monitoring for AI agents" doesn't exist as a product
2. **Learns normal behavior, flags deviations** — not rule-based, adapts to each agent type
3. **Prompt injection detection via behavioral analysis** — detects the *effect* of injection, not just the *input*
4. **Ties security + observability** — bridges your CygenIQ security background with the agent wave
5. **Production-ready integrations** — OpenTelemetry, Slack, PagerDuty, LangGraph, MCP

### Repo Structure

```
spectra/
├── README.md
├── pyproject.toml
├── src/
│   └── spectra/
│       ├── __init__.py
│       ├── instrumentation/
│       │   ├── decorator.py        # @spectra.trace decorator
│       │   ├── otel.py             # OpenTelemetry collector
│       │   ├── mcp.py              # MCP middleware
│       │   └── langgraph.py        # LangGraph integration
│       ├── profiler/
│       │   ├── profile.py          # Behavioral profile definition
│       │   ├── trainer.py          # Profile training from traces
│       │   └── markov.py           # Action sequence Markov model
│       ├── detectors/
│       │   ├── tool_anomaly.py     # Tool usage anomaly
│       │   ├── sequence_anomaly.py # Action sequence anomaly
│       │   ├── volume_anomaly.py   # Volume / duration anomaly
│       │   ├── content_anomaly.py  # Output content anomaly
│       │   └── injection.py        # Prompt injection detection
│       ├── response/
│       │   ├── policy.py           # Response policy engine
│       │   ├── alerter.py          # Alert channels (Slack, PD, etc.)
│       │   └── blocker.py          # Task blocking / quarantine
│       ├── monitor.py              # Main monitoring runtime
│       ├── dashboard/
│       │   └── app.py              # Web dashboard
│       └── cli.py
├── tests/
├── examples/
│   ├── langraph_agent.py
│   ├── mcp_agent.py
│   └── custom_agent.py
└── docs/
    ├── getting_started.md
    ├── detectors.md
    ├── integrations.md
    └── tuning.md
```

### Research References

- "AI Agents in Cybersecurity: 5 Critical Trends for 2026" (DeNexus)
- "Agentic AI Identity Security Crisis" (industry reports, 2026)
- Torq Socrates Platform — 90% Tier-1 analyst automation
- "Indirect Prompt Injection in Production LLM Applications" (OWASP, 2025)
- Z-score statistical analysis and Markov chain models — core anomaly detection methods
- OpenTelemetry specification (opentelemetry.io)
- "Behavioral Analysis for Intrusion Detection" (classic security research, adapted for agents)

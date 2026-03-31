"""spectra -- Runtime behavioral anomaly detection for AI agents.

Spectra is a lightweight observability layer that attaches to any AI agent
runtime, learns normal behavioral patterns, and flags anomalies in real-time.

Quick start::

    from spectra import Monitor, Profile
    from spectra.profiler import ProfileTrainer

    # Train a behavioral profile
    trainer = ProfileTrainer(min_traces=100)
    profile = trainer.train(agent_type="my-agent", traces=historical_traces)

    # Start monitoring
    monitor = Monitor(profile=profile, sensitivity="medium")
    monitor.start()

    # Analyze a trace
    events = await monitor.analyze(trace)
"""

from __future__ import annotations

from spectra.drift import compare as compare_profiles
from spectra.instrumentation.decorator import (
    get_current_trace,
    record_llm_call,
    record_tool_call,
    trace,
)
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    LLMCall,
    ResponseAction,
    Sensitivity,
    Severity,
    ToolCall,
)
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile as Profile
from spectra.response.alerter import (
    LogChannel,
    PagerDutyChannel,
    RateLimitedChannel,
    SlackWebhook,
    WebhookChannel,
)
from spectra.trend import Trend, TrendTracker

__version__ = "0.1.0"

__all__ = [
    "AgentTrace",
    "AnomalyEvent",
    "DetectorType",
    "LLMCall",
    "LogChannel",
    "Monitor",
    "PagerDutyChannel",
    "Profile",
    "RateLimitedChannel",
    "ResponseAction",
    "Sensitivity",
    "Severity",
    "SlackWebhook",
    "ToolCall",
    "Trend",
    "TrendTracker",
    "WebhookChannel",
    "__version__",
    "compare_profiles",
    "get_current_trace",
    "record_llm_call",
    "record_tool_call",
    "trace",
]

"""Instrumentation layer: captures agent behavior via decorators and middleware."""

from spectra.instrumentation.decorator import trace
from spectra.instrumentation.otel import OTelCollector

__all__ = ["OTelCollector", "trace"]

"""Custom exception classes for the spectra anomaly detection framework."""

from __future__ import annotations


class SpectraError(Exception):
    """Base exception for all spectra errors."""


class ProfileError(SpectraError):
    """Raised when a behavioral profile operation fails."""


class InsufficientTraceError(ProfileError):
    """Raised when there are not enough traces to train a profile."""

    def __init__(self, required: int, provided: int) -> None:
        self.required = required
        self.provided = provided
        super().__init__(
            f"Insufficient traces for training: need {required}, got {provided}"
        )


class ProfileNotTrainedError(ProfileError):
    """Raised when attempting to use a profile that has not been trained."""


class DetectorError(SpectraError):
    """Raised when an anomaly detector encounters an error."""


class AlertChannelError(SpectraError):
    """Raised when an alert channel fails to deliver."""


class MonitorError(SpectraError):
    """Raised when the monitoring runtime encounters an error."""


class MonitorNotRunningError(MonitorError):
    """Raised when an operation requires the monitor to be running."""


class ConfigurationError(SpectraError):
    """Raised when spectra is misconfigured."""

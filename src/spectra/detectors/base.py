"""Base class for all anomaly detectors."""

from __future__ import annotations

import abc
import logging

from spectra.models import AgentTrace, AnomalyEvent, DetectorType, Sensitivity
from spectra.profiler.profile import BehavioralProfile

logger = logging.getLogger(__name__)


class BaseDetector(abc.ABC):
    """Abstract base class for anomaly detectors.

    All detectors follow the same interface: given a trace and a behavioral
    profile, produce zero or more anomaly events.

    Args:
        sensitivity: Detection sensitivity preset.
    """

    detector_type: DetectorType

    def __init__(self, sensitivity: Sensitivity = Sensitivity.MEDIUM) -> None:
        self.sensitivity = sensitivity

    @abc.abstractmethod
    def analyze(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Analyze a trace against a behavioral profile for anomalies.

        Args:
            trace: The current agent execution trace.
            profile: The learned behavioral profile to compare against.

        Returns:
            List of detected anomaly events (empty if no anomalies found).
        """

    def _z_score(self, value: float, mean: float, std: float) -> float:
        """Compute the z-score of a value relative to a distribution.

        Args:
            value: Observed value.
            mean: Distribution mean.
            std: Distribution standard deviation.

        Returns:
            Absolute z-score. Returns 0.0 if std is zero and value equals
            mean, otherwise returns a large sentinel value.
        """
        if std == 0.0:
            return 0.0 if value == mean else 10.0
        return abs((value - mean) / std)

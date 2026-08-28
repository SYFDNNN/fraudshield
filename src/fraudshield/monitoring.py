"""Privacy-preserving runtime telemetry for FraudShield inference."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """One immutable process-local telemetry snapshot."""

    started_at_utc: str
    requests_total: int
    requests_failed: int
    scored_rows_total: int
    latency_milliseconds_total: float
    latency_milliseconds_average: float


class InferenceTelemetry:
    """Count requests without retaining application payloads or scores."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at_utc = datetime.now(UTC).isoformat()
        self._requests_total = 0
        self._requests_failed = 0
        self._scored_rows_total = 0
        self._latency_milliseconds_total = 0.0

    def record(
        self,
        *,
        status_code: int,
        scored_rows: int,
        latency_milliseconds: float,
    ) -> None:
        """Record aggregate metadata for one completed HTTP request."""
        if scored_rows < 0:
            raise ValueError("scored_rows must not be negative.")

        if latency_milliseconds < 0:
            raise ValueError("latency_milliseconds must not be negative.")

        with self._lock:
            self._requests_total += 1
            self._requests_failed += int(status_code >= 400)
            self._scored_rows_total += int(scored_rows)
            self._latency_milliseconds_total += float(latency_milliseconds)

    def snapshot(self) -> TelemetrySnapshot:
        """Return aggregate counters for health and demonstration purposes."""
        with self._lock:
            average_latency = (
                self._latency_milliseconds_total / self._requests_total
                if self._requests_total
                else 0.0
            )
            return TelemetrySnapshot(
                started_at_utc=self._started_at_utc,
                requests_total=self._requests_total,
                requests_failed=self._requests_failed,
                scored_rows_total=self._scored_rows_total,
                latency_milliseconds_total=self._latency_milliseconds_total,
                latency_milliseconds_average=average_latency,
            )

    def as_dict(self) -> dict[str, str | int | float]:
        """Serialize the current snapshot for the internal metrics endpoint."""
        return asdict(self.snapshot())

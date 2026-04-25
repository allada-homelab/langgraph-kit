"""Prometheus-format metrics primitives + ``/metrics`` exposition.

Hand-rolled pure-Python implementation rather than depending on
``prometheus_client``. The kit's metrics needs are bounded
(counters, gauges, histograms, label-tuple cardinality) and pulling
in a third-party library for ~200 lines of well-understood code
isn't worth the install-time tax for users who don't scrape metrics.

Format spec: https://prometheus.io/docs/instrumenting/exposition_formats/

Concurrency: this module is intended for use under a single-thread
asyncio event loop; primitives are not protected by locks. If you
later cross a thread boundary, wrap calls in a lock.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar


def _format_labels(label_names: tuple[str, ...], label_values: tuple[str, ...]) -> str:
    """Render ``{k="v",k2="v2"}`` for the Prometheus exposition format."""
    if not label_names:
        return ""
    parts = []
    for name, value in zip(label_names, label_values, strict=True):
        # Escape per spec: backslash, double-quote, newline.
        escaped = (
            str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        )
        parts.append(f'{name}="{escaped}"')
    return "{" + ",".join(parts) + "}"


class _Metric:
    """Base class for the kit's metric primitives.

    Subclasses define ``_help_kind`` (``"counter"`` / ``"gauge"`` /
    ``"histogram"``) and implement :meth:`render`.
    """

    _help_kind: ClassVar[str] = "untyped"

    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.name = name
        self.help_text = help_text
        self.label_names = tuple(labels)

    def _render_header(self) -> list[str]:
        return [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self._help_kind}",
        ]

    def render(self) -> list[str]:  # pragma: no cover — overridden
        """Render this metric to Prometheus exposition lines.

        Subclasses must implement this. Defined on the base so the
        registry's iteration over ``_Metric`` instances type-checks
        without each call site needing to narrow.
        """
        raise NotImplementedError


class Counter(_Metric):
    """Monotonically increasing counter."""

    _help_kind = "counter"

    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> None:
        super().__init__(name, help_text, labels)
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            msg = "Counter.inc requires non-negative amount"
            raise ValueError(msg)
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def value(self, **labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.label_names):
            msg = (
                f"Counter {self.name} labels mismatch: "
                f"got {sorted(labels)}, expected {sorted(self.label_names)}"
            )
            raise ValueError(msg)
        return tuple(labels[name] for name in self.label_names)

    def render(self) -> list[str]:
        lines = self._render_header()
        if not self._values:
            return lines
        for label_values, val in sorted(self._values.items()):
            lines.append(
                f"{self.name}{_format_labels(self.label_names, label_values)} {val}"
            )
        return lines


class Gauge(_Metric):
    """Value that can move up or down (currently-running counts, etc.)."""

    _help_kind = "gauge"

    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> None:
        super().__init__(name, help_text, labels)
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.label_names):
            msg = (
                f"Gauge {self.name} labels mismatch: "
                f"got {sorted(labels)}, expected {sorted(self.label_names)}"
            )
            raise ValueError(msg)
        return tuple(labels[name] for name in self.label_names)

    def render(self) -> list[str]:
        lines = self._render_header()
        for label_values, val in sorted(self._values.items()):
            lines.append(
                f"{self.name}{_format_labels(self.label_names, label_values)} {val}"
            )
        return lines


# Default histogram buckets (seconds). Reasonable for HTTP latency on a
# kit that talks to LLMs (sub-second to ~30s).
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)


class Histogram(_Metric):
    """Cumulative bucket histogram with sum + count exposition."""

    _help_kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labels: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, labels)
        self._buckets = tuple(sorted(buckets))
        # Per label-tuple state: list of cumulative counts (one per bucket
        # plus +Inf), running sum, total count.
        self._counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}
        self._totals: dict[tuple[str, ...], int] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self._buckets) + 1))
            for i, threshold in enumerate(self._buckets):
                if value <= threshold:
                    counts[i] += 1
            counts[-1] += 1  # +Inf
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._totals[key] = self._totals.get(key, 0) + 1

    def value(self, **labels: str) -> tuple[int, float]:
        """Return ``(count, sum)`` for ``labels``."""
        key = self._key(labels)
        return self._totals.get(key, 0), self._sums.get(key, 0.0)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.label_names):
            msg = (
                f"Histogram {self.name} labels mismatch: "
                f"got {sorted(labels)}, expected {sorted(self.label_names)}"
            )
            raise ValueError(msg)
        return tuple(labels[name] for name in self.label_names)

    def render(self) -> list[str]:
        lines = self._render_header()
        for label_values, counts in sorted(self._counts.items()):
            for i, threshold in enumerate(self._buckets):
                bucket_label_values = (*label_values, f"{threshold}")
                bucket_label_names = (*self.label_names, "le")
                lines.append(
                    f"{self.name}_bucket"
                    f"{_format_labels(bucket_label_names, bucket_label_values)}"
                    f" {counts[i]}"
                )
            inf_label_values = (*label_values, "+Inf")
            inf_label_names = (*self.label_names, "le")
            lines.append(
                f"{self.name}_bucket"
                f"{_format_labels(inf_label_names, inf_label_values)}"
                f" {counts[-1]}"
            )
            lines.append(
                f"{self.name}_count"
                f"{_format_labels(self.label_names, label_values)}"
                f" {self._totals[label_values]}"
            )
            lines.append(
                f"{self.name}_sum"
                f"{_format_labels(self.label_names, label_values)}"
                f" {self._sums[label_values]}"
            )
        return lines


class MetricsRegistry:
    """Holds metric primitives and renders the Prometheus exposition format."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics: dict[str, _Metric] = {}

    def register(self, metric: _Metric) -> None:
        if metric.name in self._metrics:
            msg = f"Metric already registered: {metric.name}"
            raise ValueError(msg)
        self._metrics[metric.name] = metric

    def render(self) -> str:
        """Render all metrics as the Prometheus text-exposition format."""
        out: list[str] = []
        for name in sorted(self._metrics):
            out.extend(self._metrics[name].render())
        return "\n".join(out) + "\n"

    def clear(self) -> None:
        """Drop all registered metrics. Tests use this to isolate state."""
        self._metrics.clear()


# Default registry singleton — most callers should use this.
DEFAULT_REGISTRY: MetricsRegistry = MetricsRegistry()


def render_default() -> str:
    """Convenience: render the default registry to its text exposition."""
    return DEFAULT_REGISTRY.render()


# ---------------------------------------------------------------------------
# ASGI endpoint
# ---------------------------------------------------------------------------


class MetricsEndpoint:
    """Minimal ASGI app exposing the default registry at any path it's mounted at.

    Use directly via Starlette / FastAPI ``app.add_route("/metrics", endpoint)``
    or as an ASGI sub-app.
    """

    CONTENT_TYPE: str = "text/plain; version=0.0.4; charset=utf-8"

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry or DEFAULT_REGISTRY

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:  # noqa: ARG002
        if scope.get("type") != "http":
            return  # not us
        body = self._registry.render().encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", self.CONTENT_TYPE.encode("ascii")),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

"""Coverage — Prometheus-format metrics primitives + endpoint.

Issue #28 lands the foundation: pure-Python `Counter`, `Gauge`,
`Histogram`, `MetricsRegistry`, and an ASGI endpoint that renders
the registry as Prometheus text-exposition format. No third-party
dep added; ~200 lines for ~12 features.

Wiring kit-internal counters (LLM tokens, tool calls, compactions,
rate-limit hits, HITL interrupts) is deferred to a follow-up — those
touch many modules and merit their own scope.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.observability_metrics import (
    DEFAULT_BUCKETS,
    Counter,
    Gauge,
    Histogram,
    MetricsEndpoint,
    MetricsRegistry,
)

# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


def test_counter_starts_at_zero_and_increments() -> None:
    c = Counter("hits", "test counter", labels=("kind",))
    assert c.value(kind="get") == 0.0
    c.inc(kind="get")
    c.inc(2.5, kind="get")
    assert c.value(kind="get") == 3.5


def test_counter_separate_label_buckets() -> None:
    c = Counter("hits", "test counter", labels=("kind",))
    c.inc(kind="get")
    c.inc(kind="post")
    c.inc(kind="post")
    assert c.value(kind="get") == 1.0
    assert c.value(kind="post") == 2.0


def test_counter_rejects_negative_increment() -> None:
    c = Counter("hits", "test counter")
    with pytest.raises(ValueError, match="non-negative"):
        c.inc(-1.0)


def test_counter_label_mismatch_raises() -> None:
    c = Counter("hits", "test counter", labels=("kind",))
    with pytest.raises(ValueError, match="labels mismatch"):
        c.inc(other="x")
    with pytest.raises(ValueError, match="labels mismatch"):
        c.inc()


def test_counter_render_includes_help_and_type() -> None:
    c = Counter("hits", "agent invocations", labels=("kind",))
    c.inc(kind="get")
    out = "\n".join(c.render())
    assert "# HELP hits agent invocations" in out
    assert "# TYPE hits counter" in out
    assert 'hits{kind="get"} 1.0' in out


def test_counter_render_with_no_observations_emits_only_metadata() -> None:
    c = Counter("hits", "test counter")
    out = c.render()
    assert any("# HELP" in line for line in out)
    assert any("# TYPE" in line for line in out)
    # No data lines.
    assert all(line.startswith("#") for line in out)


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------


def test_gauge_set_inc_dec() -> None:
    g = Gauge("active", "active runs")
    g.set(5.0)
    assert g.value() == 5.0
    g.inc(2.0)
    assert g.value() == 7.0
    g.dec(3.0)
    assert g.value() == 4.0


def test_gauge_with_labels() -> None:
    g = Gauge("active", "active runs", labels=("agent_id",))
    g.set(2.0, agent_id="coding")
    g.set(5.0, agent_id="qa")
    assert g.value(agent_id="coding") == 2.0
    assert g.value(agent_id="qa") == 5.0


def test_gauge_renders_as_gauge_type() -> None:
    g = Gauge("active", "active runs")
    g.set(3.0)
    out = "\n".join(g.render())
    assert "# TYPE active gauge" in out
    assert "active 3.0" in out


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def test_histogram_observes_into_buckets() -> None:
    h = Histogram("lat", "latency seconds")
    h.observe(0.05)
    h.observe(0.1)
    h.observe(0.5)
    count, total = h.value()
    assert count == 3
    assert abs(total - 0.65) < 1e-9


def test_histogram_render_emits_bucket_lines_count_and_sum() -> None:
    h = Histogram("lat", "latency seconds", buckets=(0.1, 1.0))
    h.observe(0.05)
    h.observe(0.5)
    out = "\n".join(h.render())
    # Buckets are cumulative: le=0.1 sees 1 (the 0.05 sample), le=1.0
    # sees 2 (both), +Inf also 2.
    assert 'lat_bucket{le="0.1"} 1' in out
    assert 'lat_bucket{le="1.0"} 2' in out
    assert 'lat_bucket{le="+Inf"} 2' in out
    assert "lat_count 2" in out
    assert "lat_sum 0.55" in out


def test_histogram_with_labels_partitions_observations() -> None:
    h = Histogram("lat", "latency", labels=("agent_id",), buckets=(0.5,))
    h.observe(0.1, agent_id="a")
    h.observe(0.7, agent_id="b")
    a_count, a_sum = h.value(agent_id="a")
    b_count, b_sum = h.value(agent_id="b")
    assert a_count == 1
    assert abs(a_sum - 0.1) < 1e-9
    assert b_count == 1
    assert abs(b_sum - 0.7) < 1e-9


def test_default_buckets_cover_sub_second_through_30s() -> None:
    """Sanity check the published defaults — kit talks to LLMs,
    so latencies should span ~5ms to ~30s comfortably."""
    assert min(DEFAULT_BUCKETS) <= 0.005
    assert max(DEFAULT_BUCKETS) >= 30.0
    # Strictly increasing.
    assert list(DEFAULT_BUCKETS) == sorted(DEFAULT_BUCKETS)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_render_orders_metrics_by_name() -> None:
    reg = MetricsRegistry()
    a = Counter("a_hits", "a")
    z = Counter("z_hits", "z")
    reg.register(z)
    reg.register(a)
    a.inc()
    z.inc()
    rendered = reg.render()
    assert rendered.index("a_hits") < rendered.index("z_hits")


def test_registry_rejects_duplicate_registration() -> None:
    reg = MetricsRegistry()
    reg.register(Counter("hits", "h"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Counter("hits", "h"))


def test_registry_clear_drops_metrics() -> None:
    reg = MetricsRegistry()
    reg.register(Counter("hits", "h"))
    reg.clear()
    # Re-registration after clear is fine.
    reg.register(Counter("hits", "h"))


def test_registry_render_ends_with_newline() -> None:
    """Prometheus parsers tolerate either, but a trailing newline is
    the convention — and unit tests assert it so a refactor that
    drops it doesn't surprise users."""
    reg = MetricsRegistry()
    reg.register(Counter("hits", "h"))
    out = reg.render()
    assert out.endswith("\n")


# ---------------------------------------------------------------------------
# ASGI endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_returns_200_with_text_plain_content_type() -> None:
    reg = MetricsRegistry()
    reg.register(Counter("hits", "hits", labels=("kind",)))
    reg._metrics["hits"]._values = {("get",): 7.0}  # type: ignore[attr-defined]
    endpoint = MetricsEndpoint(reg)

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await endpoint({"type": "http", "path": "/metrics"}, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200
    headers = dict(start["headers"])
    assert headers[b"content-type"].decode().startswith("text/plain")
    body = next(m for m in sent if m["type"] == "http.response.body")["body"]
    assert b'hits{kind="get"} 7.0' in body


@pytest.mark.asyncio
async def test_endpoint_skips_non_http_scopes() -> None:
    """Lifespan / websocket scopes shouldn't be answered by /metrics."""
    reg = MetricsRegistry()
    endpoint = MetricsEndpoint(reg)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(m: dict[str, Any]) -> None:
        sent.append(m)

    await endpoint({"type": "lifespan"}, receive, send)
    assert sent == []

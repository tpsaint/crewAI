"""Native OpenTelemetry instrumentation surface for crewAI.

This module exposes a thin wrapper over the OpenTelemetry **API** (not SDK).
crewAI emits spans through :func:`operation` for kickoffs, tasks, agents,
tools, LLM calls, memory, knowledge, MCP, and A2A delegation.  When no
``TracerProvider`` has been installed, the API resolves to a NoOp tracer
and spans are silently dropped (~80ns overhead per ``with`` block).

Users opt into recording by installing an OTel SDK ``TracerProvider`` in
their own process; crewAI never sets the global provider itself for the
spans emitted by this module.  See ``docs/observability/index.mdx`` for
the public guidance.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import (
    Link,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)


_TRACER_NAME = "crewai"


def _tracer() -> trace.Tracer:
    """Resolve the crewAI tracer from the current global provider.

    Always re-resolves so user code that installs a TracerProvider after
    crewAI is imported still gets recording spans.
    """
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def operation(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    links: list[Link] | None = None,
) -> Iterator[Span]:
    """Open a span around an operation, recording exceptions automatically.

    The returned context manager yields the active :class:`Span`.  Any
    exception that escapes the block sets the span status to ``ERROR``
    and records the exception event, then re-raises.

    Args:
        name: Span name (e.g. ``"execute crew"``).  Follow the
            ``"<verb> <subject>"`` convention used elsewhere in this module.
        attributes: Optional dict of attributes to set on span start.
            Keys should follow the ``crewai.<component>.<field>`` pattern.
        links: Optional list of :class:`Link` references.  Used for
            HITL resume to relate the resumed trace back to the paused one
            via :func:`follows_from`.

    Yields:
        The active :class:`Span`.  Callers may attach additional
        attributes or events to it as the operation progresses.
    """
    with _tracer().start_as_current_span(
        name,
        attributes=attributes or {},
        links=links or [],
    ) as span:
        try:
            yield span
        except BaseException as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def follows_from(trace_id: int, span_id: int) -> Link:
    """Build a FOLLOWS_FROM-style :class:`Link` for HITL resume continuity.

    OTel does not have a first-class FOLLOWS_FROM relationship kind in the
    Python SDK, so we emit a regular :class:`Link` tagged with
    ``crewai.link.type = "follows_from"``.  Backends that care about the
    distinction can filter on the attribute.

    Args:
        trace_id: Trace ID of the paused operation's span.
        span_id: Span ID of the paused operation's span.

    Returns:
        A :class:`Link` carrying a remote :class:`SpanContext` for the
        paused span, suitable to pass via the ``links=`` kwarg of
        :func:`operation`.
    """
    span_ctx = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return Link(span_ctx, attributes={"crewai.link.type": "follows_from"})


__all__ = ["follows_from", "operation"]

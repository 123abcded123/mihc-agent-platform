
from __future__ import annotations

import logging

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

_tracer = None
_initialized = False

def init_otel(config) -> None:
    global _tracer, _initialized
    if _initialized or not config.observability.otel_enabled:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

    resource = Resource.create({"service.name": config.observability.otel_service_name})
    provider = TracerProvider(resource=resource)

    if config.observability.otel_exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=config.observability.otel_endpoint)))
            logger.info("OTel exporter: otlp @ %s", config.observability.otel_endpoint)
        except Exception as exc:
            logger.warning("OTLP exporter init failed (%s), fallback to console", exc)
            provider.add_span_processor(BatchSpanProcessor(LoggingSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(LoggingSpanExporter()))
        logger.info("OTel exporter: console(logging)")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(config.observability.otel_service_name)
    _initialized = True

class LoggingSpanExporter(SpanExporter):

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            if not isinstance(span, ReadableSpan):
                continue
            lat = span.end_time - span.start_time
            status = span.status.status_code.name
            logger.info("[span] %s | %s | %dms | %s",
                        span.name, status, int(lat / 1e6), dict(span.attributes or {}))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

def get_tracer():
    global _tracer
    if _tracer is None:
        from opentelemetry import trace
        _tracer = trace.get_tracer("mihc-agent-platform")
    return _tracer

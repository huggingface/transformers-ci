from __future__ import annotations

from transformersci.otel import _export


def test_build_primary_exporter_none_without_endpoint() -> None:
    assert _export.build_primary_exporter({}) is None


def test_build_staging_exporter_none_without_endpoint() -> None:
    assert _export.build_staging_exporter({"OTEL_EXPORTER_OTLP_ENDPOINT": "x"}) is None


def test_build_primary_exporter_http_appends_signal_path() -> None:
    exporter = _export.build_primary_exporter(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        }
    )
    assert "http" in type(exporter).__module__
    assert exporter._endpoint == "http://collector:4318/v1/traces"


def test_build_primary_exporter_defaults_to_grpc() -> None:
    exporter = _export.build_primary_exporter(
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "collector:4317"}
    )
    assert "grpc" in type(exporter).__module__


def test_build_primary_exporter_prefers_traces_endpoint_and_headers() -> None:
    exporter = _export.build_primary_exporter(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://ignored:4318",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://traces:4318",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "Authorization=Bearer t",
        }
    )
    assert exporter._endpoint == "http://traces:4318/v1/traces"
    # HTTP transport keeps header keys as-is (only gRPC must lowercase them).
    assert dict(exporter._headers) == {"Authorization": "Bearer t"}


def test_build_staging_exporter_disabled_by_killswitch() -> None:
    # The hardcoded kill-switch wins regardless of a configured staging endpoint.
    assert _export.STAGING_EXPORT_DISABLED is True
    assert (
        _export.build_staging_exporter(
            {"TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT": "10.90.52.50:5317"}
        )
        is None
    )


def test_build_staging_exporter_falls_back_to_primary_protocol_and_headers(
    monkeypatch,
) -> None:
    # Flip the kill-switch off to exercise the (still-present) staging machinery.
    monkeypatch.setattr(_export, "STAGING_EXPORT_DISABLED", False)
    exporter = _export.build_staging_exporter(
        {
            "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT": "10.90.52.50:5317",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer x",
        }
    )
    # gRPC transport, with the capitalized primary header lowercased.
    assert "grpc" in type(exporter).__module__
    assert dict(exporter._headers) == {"authorization": "Bearer x"}

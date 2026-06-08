from __future__ import annotations

from pathlib import Path

from transformersci.otel import resource_plugin


class StubConfig:
    def __init__(self, option_value: str | None) -> None:
        self.option_value = option_value

    def getoption(self, name: str, default: str | None = None) -> str | None:
        assert name == "resource_metrics_file"
        return self.option_value if self.option_value is not None else default


def test_metrics_file_path_prefers_pytest_option(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_RESOURCE_METRICS_FILE", "env.jsonl")
    path = resource_plugin.metrics_file_path(StubConfig("option.jsonl"))
    assert path == Path("option.jsonl")


def test_metrics_file_path_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_RESOURCE_METRICS_FILE", "env.jsonl")
    path = resource_plugin.metrics_file_path(StubConfig(None))
    assert path == Path("env.jsonl")


def test_split_pytest_nodeid_extracts_expected_parts() -> None:
    parts = resource_plugin.split_pytest_nodeid(
        "tests/test_demo_workload.py::TestDemoWorkload::test_slow_path"
    )
    assert parts == {
        "test_class": "TestDemoWorkload",
        "test_function": "test_slow_path",
        "test_module": "test_demo_workload.py",
    }


def test_parse_otlp_headers_parses_pairs() -> None:
    assert resource_plugin._parse_otlp_headers("Authorization=Bearer abc,foo=bar") == {
        "Authorization": "Bearer abc",
        "foo": "bar",
    }


def test_parse_otlp_headers_returns_none_when_empty() -> None:
    assert resource_plugin._parse_otlp_headers(None) is None
    assert resource_plugin._parse_otlp_headers("") is None
    assert resource_plugin._parse_otlp_headers("garbage-no-equals") is None


def test_build_staging_exporter_grpc_strips_scheme_and_is_insecure() -> None:
    exporter = resource_plugin._build_staging_exporter(
        "http://10.90.52.50:4317", "grpc", {"Authorization": "Bearer x"}
    )
    assert "grpc" in type(exporter).__module__
    assert exporter._endpoint == "10.90.52.50:4317"


def test_build_staging_exporter_http_appends_signal_path() -> None:
    exporter = resource_plugin._build_staging_exporter(
        "http://10.90.52.50:4318", "http/protobuf", None
    )
    assert "http" in type(exporter).__module__
    assert exporter._endpoint == "http://10.90.52.50:4318/v1/traces"


def test_install_staging_span_processor_noop_without_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", raising=False)
    # Should simply return without touching any tracer provider.
    resource_plugin._install_staging_span_processor()


def test_install_staging_span_processor_attaches_processor(monkeypatch) -> None:
    from opentelemetry import trace

    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://10.90.52.50:4317"
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

    added: list[object] = []
    provider = _StubProvider(added.append)
    # The function does `from opentelemetry import trace` internally, so patch
    # the real module's resolver rather than a local stub.
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

    resource_plugin._install_staging_span_processor()

    assert len(added) == 1
    exporter = added[0].span_exporter  # type: ignore[attr-defined]
    assert exporter._endpoint == "10.90.52.50:4317"


def test_install_staging_span_processor_skips_without_sdk_provider(monkeypatch) -> None:
    from opentelemetry import trace

    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://10.90.52.50:4317"
    )

    # A provider without add_span_processor (like the no-export ProxyTracerProvider)
    # must be left untouched, not crash.
    monkeypatch.setattr(trace, "get_tracer_provider", object)
    resource_plugin._install_staging_span_processor()


class _StubProvider:
    """SDK-like tracer provider that records added span processors."""

    def __init__(self, recorder) -> None:
        self.add_span_processor = recorder

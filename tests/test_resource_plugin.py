from __future__ import annotations

from pathlib import Path

import pytest

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


def test_build_staging_exporter_grpc_lowercases_header_keys() -> None:
    # gRPC metadata keys must be lowercase: a capitalized "Authorization"
    # (valid over HTTP, inherited from the primary OTEL_EXPORTER_OTLP_HEADERS)
    # is rejected by gRPC core with "Illegal header key" and the whole staging
    # export fails. The builder must normalize keys for the gRPC transport.
    exporter = resource_plugin._build_staging_exporter(
        "10.90.52.50:5317", "grpc", {"Authorization": "Bearer x", "X-Foo": "y"}
    )
    assert dict(exporter._headers) == {"authorization": "Bearer x", "x-foo": "y"}


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


def test_install_staging_span_processor_killswitch_skips(monkeypatch) -> None:
    from opentelemetry import trace

    # Kill-switch ON (the default): never attach the mirror, even with an endpoint.
    assert resource_plugin.STAGING_EXPORT_DISABLED is True
    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://10.90.52.50:4317"
    )
    added: list[object] = []
    monkeypatch.setattr(
        trace, "get_tracer_provider", lambda: _StubProvider(added.append)
    )
    resource_plugin._install_staging_span_processor()
    assert added == []  # nothing attached


def test_install_staging_span_processor_attaches_processor(monkeypatch) -> None:
    from opentelemetry import trace

    monkeypatch.setattr(resource_plugin, "STAGING_EXPORT_DISABLED", False)
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


def test_install_staging_span_processor_uses_staging_protocol_override(
    monkeypatch,
) -> None:
    from opentelemetry import trace

    monkeypatch.setattr(resource_plugin, "STAGING_EXPORT_DISABLED", False)
    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://10.90.52.50:4318"
    )
    # Primary is grpc, but staging overrides to http/protobuf.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("TRANSFORMERS_TEST_OTEL_STAGING_PROTOCOL", "http/protobuf")

    added: list[object] = []
    provider = _StubProvider(added.append)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

    resource_plugin._install_staging_span_processor()

    assert len(added) == 1
    exporter = added[0].span_exporter  # type: ignore[attr-defined]
    assert "http" in type(exporter).__module__
    assert exporter._endpoint == "http://10.90.52.50:4318/v1/traces"


def test_install_staging_span_processor_swallows_build_errors(
    monkeypatch, capsys
) -> None:
    from opentelemetry import trace

    monkeypatch.setattr(resource_plugin, "STAGING_EXPORT_DISABLED", False)
    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://10.90.52.50:5317"
    )

    added: list[object] = []
    provider = _StubProvider(added.append)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

    def boom(*_args, **_kwargs):
        raise RuntimeError("staging box unreachable")

    monkeypatch.setattr(resource_plugin, "_build_staging_exporter", boom)

    # Must not raise — a staging failure cannot break the prod export / test run.
    resource_plugin._install_staging_span_processor()

    assert added == []  # nothing attached
    assert "OTEL STAGING WARNING" in capsys.readouterr().err


def test_install_staging_span_processor_skips_without_sdk_provider(monkeypatch) -> None:
    from opentelemetry import trace

    monkeypatch.setattr(resource_plugin, "STAGING_EXPORT_DISABLED", False)
    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://10.90.52.50:4317"
    )

    # A provider without add_span_processor (like the no-export ProxyTracerProvider)
    # must be left untouched, not crash.
    monkeypatch.setattr(trace, "get_tracer_provider", object)
    resource_plugin._install_staging_span_processor()


def _make_report(nodeid: str, when: str, duration: float) -> pytest.TestReport:
    # Minimal pytest TestReport as it would arrive on the controller from an
    # xdist worker.  We set duration explicitly because the constructor does not
    # accept it as a keyword argument — pytest populates it after the fact from
    # timing data collected on the worker.
    report = pytest.TestReport(
        nodeid=nodeid,
        location=("", None, nodeid),
        keywords={},
        outcome="passed",
        longrepr=None,
        when=when,
    )
    report.duration = duration
    return report


def test_runtest_logreport_sets_worker_duration_on_span() -> None:
    """Verify that the three phase durations are summed and written to the span.

    pytest_runtest_logreport is called once per phase (setup/call/teardown) with
    a report whose duration was measured on the xdist worker.  The hook should
    accumulate those three values and, after the teardown phase, set
    pytest.worker_duration_seconds on the currently open OTEL span so dashboards
    can use the real execution time instead of the inflated controller-side span
    duration.  The per-nodeid accumulator entry must also be removed once the
    attribute has been stamped.
    """
    from unittest.mock import patch

    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    nodeid = "tests/test_foo.py::test_bar"

    # Patch only for the duration of our direct calls so pytest-opentelemetry's
    # own pytest_runtest_logreport hook (fired for *this* test) is unaffected.
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "setup", 0.1))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "call", 0.5))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "teardown", 0.05))

    assert attributes.get("pytest.worker_duration_seconds") == pytest.approx(0.65)
    # Entry must be cleaned up after teardown.
    assert nodeid not in resource_plugin._worker_durations


def test_runtest_logreport_noop_when_span_not_recording() -> None:
    """Verify that the hook does nothing when no OTEL span is active.

    When OTEL is not configured, or when the hook fires outside a
    pytest_runtest_protocol hookwrapper (e.g. in non-xdist collection phases),
    trace.get_current_span() returns a non-recording no-op span.  The hook must
    silently skip set_attribute in that case rather than crashing.
    """
    from unittest.mock import patch

    class _NonRecordingSpan:
        def is_recording(self):
            return False

        def set_attribute(self, key, value):
            raise AssertionError("should not be called")

    resource_plugin._worker_durations.clear()
    nodeid = "tests/test_foo.py::test_baz"

    with patch("opentelemetry.trace.get_current_span", return_value=_NonRecordingSpan()):
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "setup", 0.1))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "call", 0.2))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "teardown", 0.05))
    # No AssertionError means set_attribute was never called — test passes.


class _StubProvider:
    """SDK-like tracer provider that records added span processors."""

    def __init__(self, recorder) -> None:
        self.add_span_processor = recorder

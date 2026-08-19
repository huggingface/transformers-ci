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

    # Defined outside the patch block and modify the set_attribute behavior of
    # `_StubSpan` below so we can inspect the attribute `pytest.worker_duration_seconds`
    # after the context exits.
    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    nodeid = "tests/test_foo.py::test_bar"

    # Use a context manager instead of monkeypatch so the stub is only active
    # during our direct calls and does not leak into other places that call
    # trace.get_current_span() (e.g. pytest-opentelemetry's own hooks).
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "setup", 0.1))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "call", 0.5))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "teardown", 0.05))

    assert attributes.get("pytest.worker_duration_seconds") == pytest.approx(0.65)
    # Entry must be cleaned up after teardown.
    assert nodeid not in resource_plugin._worker_durations


def test_runtest_logreport_noop_when_span_not_recording() -> None:
    """Verify that the hook does nothing when no OTEL span is active.

    trace.get_current_span() returns a non-recording span when OTEL env vars are
    not set, or when pytest_runtest_logreport fires outside a
    pytest_runtest_protocol hookwrapper (e.g. collection failures).  The hook
    must silently skip set_attribute in that case rather than crashing.
    """
    from unittest.mock import patch

    class _NonRecordingSpan:
        def is_recording(self):
            return False

        def set_attribute(self, key, value):
            raise AssertionError("should not be called")

    resource_plugin._worker_durations.clear()
    nodeid = "tests/test_foo.py::test_baz"

    with patch(
        "opentelemetry.trace.get_current_span", return_value=_NonRecordingSpan()
    ):
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "setup", 0.1))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "call", 0.2))
        resource_plugin.pytest_runtest_logreport(_make_report(nodeid, "teardown", 0.05))
    # No AssertionError means set_attribute was never called — test passes.


class _StubProvider:
    """SDK-like tracer provider that records added span processors."""

    def __init__(self, recorder) -> None:
        self.add_span_processor = recorder


class GcProbeConfig:
    """Config stub for the gc-probe option (the other stub asserts on its own
    option name, so this one is separate on purpose)."""

    def __init__(self, value: bool) -> None:
        self.value = value

    def getoption(self, name: str, default: bool = False) -> bool:
        assert name == "resource_gc_probe"
        return self.value


def _fake_cuda_sampler(readings: list[int], start: int, gc_probe: bool = False):
    """A sampler whose CUDA readings are scripted, so the delta arithmetic can be
    tested without a GPU."""
    sampler = resource_plugin.ResourceSampler(gc_probe=gc_probe)
    sampler.cuda_available = True
    sampler.start_cuda_allocated_bytes = start
    sampler.peak_cuda_allocated_bytes = start
    queue = list(readings)
    sampler._cuda_allocated_bytes = lambda: queue.pop(0)  # type: ignore[method-assign]
    return sampler


def test_cuda_delta_reports_what_the_next_test_inherits() -> None:
    # A test that ends holding 14 GiB more than it started with is what makes the
    # following test in the same process OOM asking for tens of MiB.
    gib = 1024**3
    sampler = _fake_cuda_sampler([15 * gib], start=1 * gib)
    metrics = sampler.stop()
    assert metrics["cuda_delta_bytes"] == 14 * gib
    assert metrics["cuda_start_allocated_bytes"] == 1 * gib
    assert metrics["cuda_end_allocated_bytes"] == 15 * gib
    # Off by default: no gc probe, no extra keys.
    assert "cuda_delta_after_gc_bytes" not in metrics


def test_cuda_delta_is_negative_when_a_test_frees_more_than_it_took() -> None:
    sampler = _fake_cuda_sampler([0], start=4096)
    assert sampler.stop()["cuda_delta_bytes"] == -4096


def test_gc_probe_separates_uncollected_garbage_from_a_live_reference() -> None:
    gib = 1024**3
    # Raw delta is 14 GiB, but a gc.collect() releases it: uncollected garbage,
    # which a `cleanup(torch_device, gc_collect=True)` in tearDown would fix.
    collectable = _fake_cuda_sampler([15 * gib, 1 * gib], start=1 * gib, gc_probe=True)
    metrics = collectable.stop()
    assert metrics["cuda_delta_bytes"] == 14 * gib
    assert metrics["cuda_delta_after_gc_bytes"] == 0

    # Still held after collecting: something owns a reference, so no tearDown
    # cleanup can free it.
    pinned = _fake_cuda_sampler([15 * gib, 15 * gib], start=1 * gib, gc_probe=True)
    assert pinned.stop()["cuda_delta_after_gc_bytes"] == 14 * gib


def test_gc_probe_is_skipped_without_cuda() -> None:
    sampler = resource_plugin.ResourceSampler(gc_probe=True)
    sampler.cuda_available = False
    metrics = sampler.stop()
    assert "cuda_delta_after_gc_bytes" not in metrics
    assert metrics["cuda_delta_bytes"] == 0


def test_gc_probe_enabled_from_flag_or_env(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_RESOURCE_GC_PROBE", raising=False)
    assert resource_plugin.gc_probe_enabled(GcProbeConfig(True)) is True
    assert resource_plugin.gc_probe_enabled(GcProbeConfig(False)) is False
    monkeypatch.setenv("PYTEST_RESOURCE_GC_PROBE", "1")
    assert resource_plugin.gc_probe_enabled(GcProbeConfig(False)) is True
    assert resource_plugin.gc_probe_enabled(None) is True
    monkeypatch.setenv("PYTEST_RESOURCE_GC_PROBE", "0")
    assert resource_plugin.gc_probe_enabled(None) is False


def _report_with_retained(nodeid: str, retained: int | None) -> "pytest.TestReport":
    """A real teardown report as it arrives on the controller, carrying the
    worker's measurement in user_properties (the channel xdist serialises)."""
    report = _make_report(nodeid, "teardown", 0.05)
    if retained is not None:
        report.user_properties.append((resource_plugin.CUDA_DELTA_PROPERTY, retained))
    return report


def test_retained_memory_is_stamped_on_the_span_from_the_report() -> None:
    """The controller stamps the span, but the measurement happened in the worker:
    under xdist the controller process holds no CUDA memory of its own, so the
    number has to arrive on the report."""
    from unittest.mock import patch

    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    gib = 1024**3
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(
            _report_with_retained("t.py::A::test_leaks", 14 * gib)
        )

    assert attributes["pytest.cuda_delta_bytes"] == 14 * gib


def test_absolute_readings_ride_the_span_beside_the_delta() -> None:
    """The delta alone is ambiguous: -17.5 GiB is equally "inherited 17.5, ended
    at 0" and "inherited 20, ended at 2.5". Both sides of the subtraction are
    stamped so the reader can tell those apart — and so "what did this test
    inherit" becomes answerable at all."""
    from unittest.mock import patch

    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    gib = 1024**3
    report = _make_report("t.py::A::test_victim", "teardown", 0.05)
    # The prod shape (gpt_oss::test_model_outputs_02): started on a card an
    # earlier test had filled, allocated nothing itself, ended with it released.
    report.user_properties.append((resource_plugin.CUDA_DELTA_PROPERTY, -17 * gib))
    report.user_properties.append((resource_plugin.CUDA_INHERITED_PROPERTY, 17 * gib))
    report.user_properties.append((resource_plugin.CUDA_RETAINED_PROPERTY, 0))
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(report)

    assert attributes["pytest.cuda_delta_bytes"] == -17 * gib
    assert attributes["pytest.cuda_inherited_bytes"] == 17 * gib
    # The absolute residual is what "left for the next test" means, and unlike
    # the delta it cannot go negative.
    assert attributes["pytest.cuda_retained_bytes"] == 0


def test_old_reports_without_the_absolute_readings_still_stamp_the_delta() -> None:
    # Backward compatibility in the other direction: a worker running an older
    # plugin sends only the delta, and that must keep working unchanged.
    from unittest.mock import patch

    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    gib = 1024**3
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(
            _report_with_retained("t.py::A::test_leaks", 3 * gib)
        )

    assert attributes["pytest.cuda_delta_bytes"] == 3 * gib
    assert "pytest.cuda_inherited_bytes" not in attributes
    assert "pytest.cuda_retained_bytes" not in attributes


def test_no_retained_attribute_when_the_report_carries_no_measurement() -> None:
    # CPU jobs and torch-less runs must stamp nothing, so the exporter emits no
    # series rather than a misleading zero.
    from unittest.mock import patch

    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(
            _report_with_retained("t.py::A::test_cpu", None)
        )

    assert "pytest.cuda_delta_bytes" not in attributes
    # Proves the absence is a decision, not an early bail: the sibling attribute
    # on the same span was still stamped.
    assert "pytest.worker_duration_seconds" in attributes


class _StubItem:
    """A real pytest.Item carries `.config`; the makereport hook reads it to
    decide whether the gc probe runs."""

    def __init__(self, nodeid: str, gc_probe: bool = False) -> None:
        self.nodeid = nodeid
        self.config = GcProbeConfig(gc_probe)


class _StubCall:
    def __init__(self, when: str) -> None:
        self.when = when


class _StubOutcome:
    def __init__(self, report) -> None:
        self._report = report

    def get_result(self):
        return self._report


def _drive_makereport(item, call, outcome):
    """Drive the hookwrapper by hand: run to the yield, then hand it the outcome
    the way pluggy would."""
    gen = resource_plugin.pytest_runtest_makereport(item, call)
    next(gen)
    try:
        gen.send(outcome)
    except StopIteration:
        pass


def test_makereport_attributes_the_delta_from_setup_to_teardown(monkeypatch) -> None:
    """The worker measures across setup..teardown, so memory a fixture allocated
    is charged to the test that used it."""
    gib = 1024**3
    readings = iter([1 * gib, 15 * gib])
    monkeypatch.setattr(resource_plugin, "_cuda_allocated_now", lambda: next(readings))
    resource_plugin._cuda_at_setup.clear()

    item = _StubItem("t.py::A::test_leaks")
    setup_report = _make_report(item.nodeid, "setup", 0.1)
    _drive_makereport(item, _StubCall("setup"), _StubOutcome(setup_report))
    assert setup_report.user_properties == []  # nothing to report yet

    teardown_report = _make_report(item.nodeid, "teardown", 0.05)
    _drive_makereport(item, _StubCall("teardown"), _StubOutcome(teardown_report))
    assert teardown_report.user_properties == [
        (resource_plugin.CUDA_DELTA_PROPERTY, 14 * gib),
        # Both sides of that subtraction ride along, so the delta can be read.
        (resource_plugin.CUDA_INHERITED_PROPERTY, 1 * gib),
        (resource_plugin.CUDA_RETAINED_PROPERTY, 15 * gib),
    ]
    # The per-nodeid entry must not leak across tests.
    assert item.nodeid not in resource_plugin._cuda_at_setup


def test_makereport_records_a_release_as_a_negative_delta(monkeypatch) -> None:
    """The prod shape that made the delta unreadable (gpt_oss's
    `test_model_outputs_02`, -17.50 GiB): a test that allocates nothing itself,
    starting on a card an earlier test filled and ending with it released. The
    delta is negative; the absolute pair says plainly what happened."""
    gib = 1024**3
    readings = iter([17 * gib, 0])
    monkeypatch.setattr(resource_plugin, "_cuda_allocated_now", lambda: next(readings))
    resource_plugin._cuda_at_setup.clear()

    item = _StubItem("t.py::A::test_victim")
    _drive_makereport(
        item, _StubCall("setup"), _StubOutcome(_make_report(item.nodeid, "setup", 0.1))
    )
    teardown_report = _make_report(item.nodeid, "teardown", 0.05)
    _drive_makereport(item, _StubCall("teardown"), _StubOutcome(teardown_report))

    props = dict(teardown_report.user_properties)
    assert props[resource_plugin.CUDA_DELTA_PROPERTY] == -17 * gib
    assert props[resource_plugin.CUDA_INHERITED_PROPERTY] == 17 * gib
    # Never negative: this is what the next test actually inherits.
    assert props[resource_plugin.CUDA_RETAINED_PROPERTY] == 0


def test_makereport_reports_nothing_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(resource_plugin, "_cuda_allocated_now", lambda: None)
    resource_plugin._cuda_at_setup.clear()

    item = _StubItem("t.py::A::test_cpu")
    _drive_makereport(
        item, _StubCall("setup"), _StubOutcome(_make_report(item.nodeid, "setup", 0.1))
    )
    teardown_report = _make_report(item.nodeid, "teardown", 0.05)
    _drive_makereport(item, _StubCall("teardown"), _StubOutcome(teardown_report))
    assert teardown_report.user_properties == []


def test_makereport_reports_nothing_when_setup_was_never_measured(monkeypatch) -> None:
    # e.g. the plugin was enabled mid-session, or setup ran on a CPU-only device
    # that later gained a reading. No baseline means no claim.
    monkeypatch.setattr(resource_plugin, "_cuda_allocated_now", lambda: 5 * 1024**3)
    resource_plugin._cuda_at_setup.clear()
    item = _StubItem("t.py::A::test_orphan")
    teardown_report = _make_report(item.nodeid, "teardown", 0.05)
    _drive_makereport(item, _StubCall("teardown"), _StubOutcome(teardown_report))
    assert teardown_report.user_properties == []


def test_gc_probe_adds_the_after_gc_property_to_the_report(monkeypatch) -> None:
    """With the probe on, the worker reports BOTH numbers: what the next test
    inherits, and how much of it survives a collection."""
    gib = 1024**3
    readings = iter([1 * gib, 15 * gib, 1 * gib])  # setup, teardown, after gc
    monkeypatch.setattr(resource_plugin, "_cuda_allocated_now", lambda: next(readings))
    monkeypatch.setattr(resource_plugin, "gc_probe_enabled", lambda _cfg: True)
    resource_plugin._cuda_at_setup.clear()

    item = _StubItem("t.py::A::test_collectable")
    _drive_makereport(
        item, _StubCall("setup"), _StubOutcome(_make_report(item.nodeid, "setup", 0.1))
    )
    report = _make_report(item.nodeid, "teardown", 0.05)
    _drive_makereport(item, _StubCall("teardown"), _StubOutcome(report))

    props = dict(report.user_properties)
    # Raw delta is unaffected by the probe: it is recorded before the collect.
    assert props[resource_plugin.CUDA_DELTA_PROPERTY] == 14 * gib
    # …and the collection released all of it → a tearDown would fix this test.
    assert props[resource_plugin.CUDA_DELTA_AFTER_GC_PROPERTY] == 0


def test_gc_probe_reports_memory_that_survives_collection(monkeypatch) -> None:
    gib = 1024**3
    readings = iter([1 * gib, 15 * gib, 15 * gib])
    monkeypatch.setattr(resource_plugin, "_cuda_allocated_now", lambda: next(readings))
    monkeypatch.setattr(resource_plugin, "gc_probe_enabled", lambda _cfg: True)
    resource_plugin._cuda_at_setup.clear()

    item = _StubItem("t.py::A::test_pinned")
    _drive_makereport(
        item, _StubCall("setup"), _StubOutcome(_make_report(item.nodeid, "setup", 0.1))
    )
    report = _make_report(item.nodeid, "teardown", 0.05)
    _drive_makereport(item, _StubCall("teardown"), _StubOutcome(report))
    # Nothing was freed: a live reference, so no tearDown cleanup can help.
    assert (
        dict(report.user_properties)[resource_plugin.CUDA_DELTA_AFTER_GC_PROPERTY]
        == 14 * gib
    )


def test_after_gc_property_is_stamped_on_the_span() -> None:
    from unittest.mock import patch

    attributes: dict = {}

    class _StubSpan:
        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            attributes[key] = value

    resource_plugin._worker_durations.clear()
    report = _make_report("t.py::A::test_pinned", "teardown", 0.05)
    report.user_properties.append((resource_plugin.CUDA_DELTA_PROPERTY, 14 * 1024**3))
    report.user_properties.append(
        (resource_plugin.CUDA_DELTA_AFTER_GC_PROPERTY, 14 * 1024**3)
    )
    with patch("opentelemetry.trace.get_current_span", return_value=_StubSpan()):
        resource_plugin.pytest_runtest_logreport(report)
    assert attributes["pytest.cuda_delta_after_gc_bytes"] == 14 * 1024**3


def test_empty_env_values_mean_off(monkeypatch) -> None:
    """serge-verify-slow passes these as empty strings when its memory_probe
    input is false (`${{ inputs.memory_probe && '1' || '' }}`), so empty MUST
    read as disabled — otherwise a normal verify run would start sampling and
    collecting."""
    monkeypatch.setenv("PYTEST_RESOURCE_GC_PROBE", "")
    monkeypatch.setenv("PYTEST_RESOURCE_METRICS_FILE", "")
    monkeypatch.delenv("TRANSFORMERS_TEST_RESOURCE_METRICS_FILE", raising=False)
    assert resource_plugin.gc_probe_enabled(None) is False
    assert resource_plugin.metrics_file_path(None) is None

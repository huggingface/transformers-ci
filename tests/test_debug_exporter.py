from __future__ import annotations

from transformersci.otel import debug_exporter


class _FakeResult:
    name = "SUCCESS"


class _FailResult:
    name = "FAILURE"


def test_span_tally_separates_success_failure_and_queue_drop() -> None:
    tally = debug_exporter._SpanTally()
    tally.record_produced(10)
    tally.record_produced(3)
    # 8 spans shipped OK; 2 spans the exporter failed to ship; 3 produced spans
    # never reached an exporter at all (BSP queue overflow): 13 - (8 + 2) = 3.
    tally.record_export(8, ok=True)
    tally.record_export(2, ok=False)

    line = tally.summary_line()
    assert "produced=13" in line
    assert "submitted=10" in line
    assert "exported=8" in line
    # failed = transport loss (the signal the old blind tally hid).
    assert "failed=2" in line
    # not_exported = produced - submitted = BSP queue overflow.
    assert "not_exported=3" in line
    assert "export_calls=2" in line
    assert "failed_calls=1" in line
    assert tally.exported == 8
    assert tally.failed == 2
    assert tally.submitted == 10


def test_counting_on_end_tallies_each_span_and_delegates(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    seen = []
    wrapped = debug_exporter._make_counting_on_end(lambda self, span: seen.append(span))
    wrapped("self", "span-a")
    wrapped("self", "span-b")

    assert seen == ["span-a", "span-b"]
    assert tally.produced == 2


def test_logging_export_counts_success_as_exported(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    result = _FakeResult()
    wrapped = debug_exporter._make_logging_export(lambda self, spans: result)
    assert wrapped("self", ["s1", "s2", "s3"]) is result

    assert tally.submitted == 3
    assert tally.exported == 3
    assert tally.failed == 0
    assert tally.export_calls == 1


def test_logging_export_counts_failure_result_as_lost(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    # A FAILURE result (the SDK exhausted retries) must count as lost, not
    # exported — this is the timeout/rejection case the old tally hid.
    wrapped = debug_exporter._make_logging_export(lambda self, spans: _FailResult())
    wrapped("self", ["s1", "s2", "s3", "s4"])

    assert tally.submitted == 4
    assert tally.exported == 0
    assert tally.failed == 4
    assert tally.failed_calls == 1


def test_logging_export_counts_exception_as_lost(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    def boom(self, spans):
        raise RuntimeError("export failed")

    wrapped = debug_exporter._make_logging_export(boom)
    try:
        wrapped("self", ["s1", "s2"])
    except RuntimeError:
        pass

    # An exception means the batch never shipped — count it as failed (lost),
    # not exported. The whole point of the fix.
    assert tally.exported == 0
    assert tally.failed == 2
    assert tally.failed_calls == 1


def test_tally_keeps_prod_and_stage_separate() -> None:
    tally = debug_exporter._SpanTally()
    tally.record_produced(100, dest="prod")
    tally.record_produced(100, dest="stage")
    tally.record_export(90, ok=True, dest="prod")
    tally.record_export(10, ok=False, dest="prod")
    # Staging is much flakier — but it must NOT pollute the prod headline.
    tally.record_export(40, ok=True, dest="stage")
    tally.record_export(60, ok=False, dest="stage")

    # Back-compat properties report PROD only.
    assert tally.produced == 100
    assert tally.exported == 90
    assert tally.failed == 10

    line = tally.summary_line()
    assert "produced=100" in line
    assert "exported=90" in line
    assert "failed=10" in line  # prod failures only
    assert "stage_failed=60" in line  # staging failures kept separate
    assert "stage_exported=40" in line


def test_export_to_staging_endpoint_classified_as_stage(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)
    monkeypatch.setenv(
        "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT", "http://stagebox:4318"
    )

    class _Exp:
        def __init__(self, endpoint):
            self._endpoint = endpoint

        def export(self, spans):
            return _FakeResult()

    prod = _Exp("https://prod.example/v1/traces")
    stage = _Exp("http://stagebox:4318/v1/traces")
    debug_exporter._make_logging_export(_Exp.export)(prod, ["a", "b"])
    debug_exporter._make_logging_export(_Exp.export)(stage, ["c", "d", "e"])

    # Prod headline counts only the prod export; staging is sidelined.
    assert tally.exported == 2
    assert "stage_exported=3" in tally.summary_line()


class _FakeExporter:
    """Stand-in for an OTLPSpanExporter carrying a resolved ``_timeout``."""

    def __init__(self, timeout) -> None:
        self._timeout = timeout

    def export(self, spans):
        return _FakeResult()


def test_logging_export_line_reports_exporter_timeout(monkeypatch, capsys) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    exporter = _FakeExporter(timeout=30)
    wrapped = debug_exporter._make_logging_export(_FakeExporter.export)
    wrapped(exporter, ["s1", "s2"])

    line = capsys.readouterr().err
    # The resolved per-export timeout must show next to duration so a timeout
    # failure (duration ≈ timeout) is self-evident in the log.
    assert "timeout_s=30" in line


def test_logging_export_timeout_unknown_when_absent(monkeypatch, capsys) -> None:
    monkeypatch.setattr(debug_exporter, "_TALLY", debug_exporter._SpanTally())

    wrapped = debug_exporter._make_logging_export(lambda self, spans: _FakeResult())
    wrapped("self", ["s1"])  # plain object, no _timeout

    assert "timeout_s=?" in capsys.readouterr().err


def test_bsp_config_logged_once(monkeypatch, capsys) -> None:
    monkeypatch.setattr(debug_exporter, "_TALLY", debug_exporter._SpanTally())
    monkeypatch.setattr(debug_exporter, "_BSP_CONFIG_LOGGED", False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "30000")

    class _FakeBSP:
        max_queue_size = 2048
        max_export_batch_size = 512
        schedule_delay_millis = 5000
        export_timeout_millis = 30000

    processor = _FakeBSP()
    wrapped = debug_exporter._make_counting_on_end(lambda self, span: None)
    wrapped(processor, "span-a")
    wrapped(processor, "span-b")

    err = capsys.readouterr().err
    # Logged exactly once, and carries the batch + timeout knobs.
    assert err.count("OTEL DEBUG BSP CONFIG") == 1
    assert "max_queue_size=2048" in err
    assert "export_timeout_millis=30000" in err
    assert "otlp_timeout_env=30000" in err

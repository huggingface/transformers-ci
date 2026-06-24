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

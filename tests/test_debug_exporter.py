from __future__ import annotations

from transformersci.otel import debug_exporter


class _FakeResult:
    name = "SUCCESS"


def test_span_tally_reports_produced_exported_and_gap() -> None:
    tally = debug_exporter._SpanTally()
    tally.record_produced(10)
    tally.record_produced(3)
    tally.record_exported(8)
    tally.record_exported(2)

    line = tally.summary_line()
    assert "produced=13" in line
    assert "exported=10" in line
    # The gap (produced - exported) is the queue drop we are hunting.
    assert "not_exported=3" in line
    assert "export_calls=2" in line


def test_counting_on_end_tallies_each_span_and_delegates(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    seen = []
    wrapped = debug_exporter._make_counting_on_end(lambda self, span: seen.append(span))
    wrapped("self", "span-a")
    wrapped("self", "span-b")

    assert seen == ["span-a", "span-b"]
    assert tally.produced == 2


def test_logging_export_tallies_batch_size_and_returns_result(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    result = _FakeResult()
    wrapped = debug_exporter._make_logging_export(lambda self, spans: result)
    assert wrapped("self", ["s1", "s2", "s3"]) is result

    assert tally.exported == 3
    assert tally.export_calls == 1


def test_logging_export_still_tallies_on_exception(monkeypatch) -> None:
    tally = debug_exporter._SpanTally()
    monkeypatch.setattr(debug_exporter, "_TALLY", tally)

    def boom(self, spans):
        raise RuntimeError("export failed")

    wrapped = debug_exporter._make_logging_export(boom)
    try:
        wrapped("self", ["s1", "s2"])
    except RuntimeError:
        pass

    # Even a failed export counts what it tried to ship, so produced-vs-exported
    # stays comparable rather than blaming the queue for a transport failure.
    assert tally.exported == 2

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

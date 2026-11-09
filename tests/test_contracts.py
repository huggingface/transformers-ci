"""run-model-contracts: tmic check plus CI telemetry, the PR gate, fleets and the report."""

import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from transformersci.contracts import fleet as fleet_api
from transformersci.contracts import report, run, select_contracts
from transformersci.otel import instrument

REVISION = "d" * 40
RECORD = {
    "contract": "huggingface-mic/mic-demo",
    "contract_name": "demo",
    "contract_revision": "a" * 40,
    "framework": "transformers",
    "variant": "default",
    "target": "demo-mic",
    "kind": "fixture",
    "revision": "b" * 40,
    "platform": "cpu-x86_64",
    "seconds": 2.5,
}


@pytest.fixture
def tmic(monkeypatch):
    """A stand-in transformers_mic: fleet resolution, and a `tmic check` that prints given lines."""
    module = ModuleType("transformers_mic")
    fleet = ModuleType("transformers_mic.fleet")
    fleet.REPOSITORY = "huggingface-mic/transformers-fleet"
    fleet.resolve = lambda revision: REVISION if revision == "main" else revision
    fleet.read_pin = lambda path: Path(path).read_text().strip()
    module.fleet = fleet
    monkeypatch.setitem(sys.modules, "transformers_mic", module)
    monkeypatch.setitem(sys.modules, "transformers_mic.fleet", fleet)
    commands = []
    real = run.check_command

    def runs(stdout=(), stderr="", code=0):
        script = textwrap.dedent(
            f"""
            import sys
            for line in {list(stdout)!r}:
                print(line, flush=True)
            sys.stderr.write({stderr!r})
            sys.exit({code})
            """
        )

        def command(args, options):
            commands.append(real(args, options))
            return [sys.executable, "-c", script]

        monkeypatch.setattr(run, "check_command", command)
        return commands

    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    return runs


@pytest.fixture
def spans(monkeypatch):
    memory = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setattr(instrument, "build_primary_exporter", lambda env: memory)
    monkeypatch.setattr(instrument, "build_staging_exporter", lambda env: None)
    return memory


def test_check_command_forwards_the_selection_to_tmic():
    args = run.parser().parse_args(
        [
            "--fixtures-only",
            "--variant",
            "bfloat16",
            "--changed-files",
            "changed.txt",
            "--diff-base",
            "HEAD^1",
            "--run-declared",
            "--contract",
            "demo",
            "--target",
            "demo-mic",
        ]
    )
    command = run.check_command(
        args, ["--fleet", "control", "--fleet-revision", REVISION]
    )
    assert command[:4] == [sys.executable, "-m", "transformers_mic.cli", "check"]
    options = command[4:]
    assert options[:4] == ["--fleet", "control", "--fleet-revision", REVISION]
    for flag in ("--fixtures-only", "--run-declared"):
        assert flag in options
    for flag, value in (
        ("--integration", "transformers"),
        ("--variant", "bfloat16"),
        ("--diff-base", "HEAD^1"),
        ("--contract", "demo"),
        ("--target", "demo-mic"),
    ):
        assert options[options.index(flag) + 1] == value
    assert Path(options[options.index("--changed-files") + 1]).is_absolute()
    assert "--run-all" not in options and "--isolate" not in options
    # --diff-base means nothing without the changed files it compares.
    alone = run.parser().parse_args(["--diff-base", "HEAD^1", "--run-all"])
    options = run.check_command(alone, [])
    assert "--diff-base" not in options and "--run-all" in options


def test_records_carry_the_pinned_fleet_into_stdout_output_spans_and_report(
    tmic, spans, capsys, tmp_path
):
    failed = {**RECORD, "status": "fail", "failures": ["tokens: [9] != [1, 2]"]}
    commands = tmic(
        stdout=["contract chatter", json.dumps(failed)],
        stderr='{"selection": {}}\n',
        code=1,
    )
    output = tmp_path / "results.jsonl"
    code = run.main(
        ["--fleet", "prototypes", "--fixtures-only", "--output", str(output)]
    )
    assert code == 1
    assert commands[0][4:8] == ["--fleet", "prototypes", "--fleet-revision", REVISION]
    out = capsys.readouterr()
    assert "contract chatter" in out.out and '"selection"' in out.err
    record = json.loads(out.out.splitlines()[-1])
    assert record["fleet"] == "prototypes"
    assert record["fleet_repository"] == "huggingface-mic/transformers-fleet"
    assert record["fleet_revision"] == REVISION
    assert [json.loads(line) for line in output.read_text().splitlines()] == [record]

    finished = {s.name: s for s in spans.get_finished_spans()}
    case = finished["tests/contracts::demo::demo-mic[default]"]
    assert case.attributes["pytest.span_type"] == "test"
    assert case.attributes["pytest.nodeid"] == case.name
    assert case.attributes["mic.status"] == "fail"
    assert case.attributes["mic.fleet_revision"] == REVISION
    assert case.status.status_code.name == "ERROR"
    # Backdated by the record's seconds: the case ran in tmic's process.
    assert (case.end_time - case.start_time) / 1e9 == pytest.approx(2.5, abs=0.5)
    error = next(e for e in case.events if e.name == "exception")
    assert error.attributes["exception.type"] == "MICAssertionError"
    assert "tokens:" in error.attributes["exception.stacktrace"]
    assert "pytest.span_type" not in finished["fast-integration"].attributes
    assert "tests/contracts::runner" not in finished

    report.main([str(output)])
    summary = capsys.readouterr().out
    assert f"--fleet prototypes --fleet-revision {REVISION}" in summary
    assert "--contract demo --target demo-mic" in summary


def test_passing_and_skipped_cases_are_ok_spans(tmic, spans, capsys):
    tmic(
        stdout=[
            json.dumps({**RECORD, "status": "pass", "failures": []}),
            json.dumps(
                {
                    **RECORD,
                    "target": "other-mic",
                    "status": "unsupported",
                    "skip": "declared",
                    "failures": ["not measured"],
                    "seconds": None,
                }
            ),
        ]
    )
    assert run.main(["--fleet-revision", "e" * 40]) == 0
    cases = [
        s
        for s in spans.get_finished_spans()
        if s.attributes.get("pytest.span_type") == "test"
    ]
    assert [c.attributes["mic.status"] for c in cases] == ["pass", "unsupported"]
    assert all(c.status.status_code.name != "ERROR" for c in cases)
    assert {c.attributes["mic.fleet_revision"] for c in cases} == {"e" * 40}


def test_a_run_that_dies_before_any_case_gets_a_runner_span(tmic, spans, capsys):
    tmic(stderr="Traceback ...\nOSError: no space left\n", code=2)
    assert run.main([]) == 2
    finished = {s.name: s for s in spans.get_finished_spans()}
    runner = finished["tests/contracts::runner"]
    assert runner.status.status_code.name == "ERROR"
    error = next(e for e in runner.events if e.name == "exception")
    assert error.attributes["exception.type"] == "MICRunnerError"
    assert "no space left" in error.attributes["exception.stacktrace"]


def test_the_framework_pin_is_the_default_fleet_revision(
    tmic, monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/contracts.lock").write_text("f" * 40 + "\n")
    commands = tmic()
    run.main(["--fixtures-only"])
    assert commands[0][4:8] == ["--fleet", "control", "--fleet-revision", "f" * 40]


def test_a_custom_lock_is_passed_through_and_recorded(tmic, tmp_path, capsys):
    lock = tmp_path / "fleet.lock"
    commands = tmic(stdout=[json.dumps({**RECORD, "status": "pass", "failures": []})])
    assert run.main(["--lock", str(lock)]) == 0
    assert commands[0][4:6] == ["--lock", str(lock.resolve())]
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["lock"] == str(lock.resolve()) and "fleet_revision" not in record


def test_a_custom_lock_cannot_silently_ignore_a_dataset_revision(tmp_path):
    for argv in (
        ["--lock", str(tmp_path / "fleet.lock"), "--fleet-revision", "main"],
        ["--lock", str(tmp_path / "fleet.lock"), "--fleet", "control"],
    ):
        with pytest.raises(SystemExit) as error:
            run.main(argv)
        assert error.value.code == 2


def test_may_select_gates_on_paths_alone():
    for paths in (
        ["src/transformers/models/gpt2/modeling_gpt2.py"],
        ["src/transformers/modeling_utils.py"],
        ["pyproject.toml"],
        ["tests/contracts.lock"],
        ["tests/contracts/fleet/bert.json"],
        ["tests/contracts/expectations/bert/cpu.json"],
        ["utils/contracts/run.py"],
    ):
        assert select_contracts.may_select(paths)
    for paths in (
        [],
        [".gitattributes", "Makefile", "utils/check_gitattributes.py"],
        ["docs/source/en/index.md", "tests/models/bert/test_modeling_bert.py"],
    ):
        assert not select_contracts.may_select(paths)


def test_hub_fleet_resolves_and_downloads_an_immutable_dataset(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError
    from types import SimpleNamespace

    revision = "c" * 40
    calls = []

    class API:
        def dataset_info(self, repo, revision):
            assert repo == "huggingface-mic/transformers-fleet"
            assert revision == "main"
            return SimpleNamespace(sha="c" * 40)

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        if filename.endswith(".csv"):  # a revision from before the tables
            raise EntryNotFoundError(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert fleet_api.lock_path("prototypes") == (
        tmp_path / "prototype-fleet.lock",
        revision,
    )
    assert calls[-1] == (
        fleet_api.REPOSITORY,
        "prototype-fleet.lock",
        {"repo_type": "dataset", "revision": revision},
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: pytest.fail("immutable revision must not resolve main again"),
    )
    assert fleet_api.lock_path("control", revision)[1] == revision


def test_framework_dataset_pin_is_strict(tmp_path):
    path = tmp_path / "contracts.lock"
    path.write_text("e" * 40 + "\n")
    assert fleet_api.read_revision(path) == "e" * 40
    for value in ("main", "", "e" * 40 + "\nother"):
        path.write_text(value)
        with pytest.raises(ValueError, match="immutable dataset"):
            fleet_api.read_revision(path)


FLEET_TABLES = {
    "contracts": (
        "contract,target,mic_repo,revision,contract_sha256,fixture_sha256,checkpoints\n"
        f"SmolLM,mic-SmolLM-slim,huggingface-mic/smollm-mic,{'c' * 40},{'3' * 64},{'4' * 64},"
        f"HuggingFaceTB/SmolLM-135M@{'e' * 40}; HuggingFaceTB/SmolLM-1.7B@{'f' * 40}\n"
        f"SmolLM,mic-SmolLM-1.7B-slim,huggingface-mic/smollm-1.7b-mic,{'d' * 40},,{'5' * 64},\n"
        f"TimesFM,,huggingface-mic/timesfm-mic,{'a' * 40},{'1' * 64},{'2' * 64},google/timesfm@{'b' * 40}\n"
    ),
    "environment": "package,version\npython,3.11\ntorch,2.14.0\n",
    "known_failures": (
        "contract,target,issue,match,skip_outputs\n"
        "SmolLM,mic-SmolLM-slim,https://github.com/org/repo/issues/1,,\n"
        "TimesFM,timesfm-mic,https://github.com/org/repo/issues/2,overflow,training.loss; training.gradients\n"
    ),
}
FLEET_LOCK = {
    "framework": "transformers",
    "environment": {"python": "3.11", "torch": "2.14.0"},
    "contracts": {
        "SmolLM": {
            "integration": {
                "repo": "huggingface-mic/smollm-mic",
                "revision": "c" * 40,
                "files_sha256": "3" * 64,
            },
            "fixtures": {
                "huggingface-mic/smollm-mic": {
                    "revision": "c" * 40,
                    "files_sha256": "4" * 64,
                    "target": "mic-SmolLM-slim",
                },
                "huggingface-mic/smollm-1.7b-mic": {
                    "revision": "d" * 40,
                    "files_sha256": "5" * 64,
                    "target": "mic-SmolLM-1.7B-slim",
                },
            },
            "checkpoints": {
                "HuggingFaceTB/SmolLM-135M": "e" * 40,
                "HuggingFaceTB/SmolLM-1.7B": "f" * 40,
            },
        },
        "TimesFM": {
            "integration": {
                "repo": "huggingface-mic/timesfm-mic",
                "revision": "a" * 40,
                "files_sha256": "1" * 64,
            },
            "fixtures": {
                "huggingface-mic/timesfm-mic": {
                    "revision": "a" * 40,
                    "files_sha256": "2" * 64,
                }
            },
            "checkpoints": {"google/timesfm": "b" * 40},
        },
    },
    "known_failures": {
        "SmolLM/mic-SmolLM-slim": "https://github.com/org/repo/issues/1",
        "TimesFM/timesfm-mic": {
            "issue": "https://github.com/org/repo/issues/2",
            "match": "overflow",
            "skip_outputs": ["training.loss", "training.gradients"],
        },
    },
}


@pytest.fixture
def hub_fleet(monkeypatch, tmp_path):
    """A fake fleet dataset: {revision: {filename: text}}."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    files = {}

    def download(repo, filename, repo_type, revision):
        assert (repo, repo_type) == (fleet_api.REPOSITORY, "dataset")
        if filename not in files[revision]:
            raise EntryNotFoundError(filename)
        path = tmp_path / "hub" / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[revision][filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(fleet_api.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    return files


def tables(fleet):
    return {f"{fleet}/{name}.csv": text for name, text in FLEET_TABLES.items()}


def test_hub_fleet_reads_the_tables_into_a_valid_lock(hub_fleet):
    revision = "1" * 40
    hub_fleet[revision] = tables("prototypes")
    path, resolved = fleet_api.lock_path("prototypes", revision)
    assert resolved == revision and path.name == "prototype-fleet.lock"
    assert json.loads(path.read_text()) == FLEET_LOCK


def test_hub_fleet_tables_must_agree_with_the_transition_json(hub_fleet):
    agreeing, stale = "1" * 40, "2" * 40
    hub_fleet[agreeing] = {
        **tables("control"),
        "fleet-lock.json": json.dumps({"comment": "ignored", **FLEET_LOCK}),
    }
    assert (
        json.loads(fleet_api.lock_path("control", agreeing)[0].read_text())
        == FLEET_LOCK
    )
    # An older tmic that updated only the JSON must not be shadowed by stale tables
    hub_fleet[stale] = {
        **tables("control"),
        "fleet-lock.json": json.dumps({**FLEET_LOCK, "environment": {}}),
    }
    with pytest.raises(ValueError, match="tables and fleet-lock.json disagree"):
        fleet_api.lock_path("control", stale)


def test_hub_fleet_rejects_a_contract_without_its_carrying_row():
    broken = {
        **FLEET_TABLES,
        "contracts": FLEET_TABLES["contracts"].replace(f",{'3' * 64},", ",,"),
    }
    with pytest.raises(ValueError, match="checkpoints belong on the row carrying"):
        fleet_api.from_tables(broken)


def test_coverage_model_types_reads_mapping_and_updates():
    from transformersci.contracts.coverage import model_types

    mappings = 'CONFIG_MAPPING_NAMES = OrderedDict(\n    [\n        ("bert", "BertConfig"),\n        ("t5", "T5Config"),\n    ]\n)\n'
    auto = (
        'CONFIG_MAPPING_NAMES.update({"mimi_decoder": "MimiDecoderConfig"})\n'
        'CONFIG_MAPPING_NAMES = OrderedDict(**{"gpt-sw3": "GPT2Config"}, **CONFIG_MAPPING_NAMES)\n'
        'OTHER = OrderedDict([("ignored", "X")])\n'
    )
    assert model_types(mappings, auto) == ["bert", "gpt-sw3", "mimi_decoder", "t5"]

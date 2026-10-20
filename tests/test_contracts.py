"""Exercise MIC execution, strict assertions, selection, and real OTEL export."""

import json
import subprocess
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from transformersci.contracts import run, select_contracts
from transformersci.otel import instrument


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    contract = {
        "integration": {"repo": "test/demo-integration", "revision": "a" * 40},
        "fixtures": {"test/demo-slim": {"revision": "b" * 40}},
        "checkpoints": {},
    }
    lock = {
        "framework": "transformers",
        "contracts": {"demo": contract},
        "environment": {},
    }
    lock_path = tmp_path / "fleet.lock"
    lock_path.write_text(json.dumps(lock))
    rules = {
        "runtime": {"compare": "exact"},
        "tokens": {"compare": "exact"},
        "logits": {"compare": "allclose", "atol": 0.001, "rtol": 0.001},
        "checks": {"compare": "all_true"},
        "updated": {"compare": "true"},
    }
    manifest = {
        "mic": "0.2",
        "name": "Demo",
        "usage": "Regression test",
        "checkpoints": {"test/checkpoint": {"fixture": "test/demo-slim"}},
        "fixtures": {
            "test/demo-slim": {
                "built_from": "test/checkpoint",
                "covers": ["test/checkpoint"],
            }
        },
        "integrations": {
            "transformers": {
                "entry_point": "usage.py",
                "format": "runnable-markdown",
                "requires": [],
                "targets": ["fixtures"],
                "variants": {"default": {}},
                "training": False,
            }
        },
        "outputs": rules,
    }
    (tmp_path / "integration.json").write_text(json.dumps(manifest))
    actual = {
        "runtime": {"device": "cpu"},
        "tokens": [1, 2],
        "logits": [0.1, 0.2],
        "checks": {"finite": True},
        "updated": True,
    }
    entry = tmp_path / "usage.py"
    entry.write_text(f"import json\nprint(json.dumps({actual!r}))\n")
    baseline = tmp_path / "expectations/demo/cpu-test/demo-slim.json"
    baseline.parent.mkdir(parents=True)
    identity = {
        "contract": "test/demo-integration",
        "contract_revision": "a" * 40,
        "target": "demo-slim",
        "revision": "b" * 40,
        "framework": "transformers",
        "variant": "default",
        "environment": {},
    }
    baseline.write_text(json.dumps({"recorded_with": identity, "outputs": actual}))
    monkeypatch.setattr(run, "contract_folder", lambda *a: (tmp_path, "a" * 40))
    monkeypatch.setattr(run, "fixture_folder", lambda *a: tmp_path)
    monkeypatch.setattr(run, "fixture_problem", lambda *a: None)
    monkeypatch.setattr(run, "contract_code", lambda p: p.read_text())
    monkeypatch.setattr(run, "platform_class", lambda *a: "cpu-test")
    monkeypatch.setattr(run.Executor, "environment", lambda *a: {})
    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    return lock_path, entry, baseline, actual


def invoke(fleet, capsys, *args):
    code = run.main(["--lock", str(fleet[0]), "--fixtures-only", *args])
    lines = capsys.readouterr().out.splitlines()
    return code, [json.loads(line) for line in lines]


def test_baseline_execution_passes(fleet, capsys):
    code, rows = invoke(fleet, capsys)
    assert code == 0 and rows[0]["status"] == "pass"


def test_fixture_provenance_is_in_results(fleet, capsys):
    build = {"built_from": {"repo": "creator/Large-Demo"}, "weights": "random"}
    (fleet[0].parent / "build.json").write_text(json.dumps(build))
    code, rows = invoke(fleet, capsys)
    assert code == 0
    assert rows[0]["source_model"] == "creator/Large-Demo"
    assert rows[0]["fixture_repo"] == "test/demo-slim"
    assert rows[0]["weights"] == "random"


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"tokens": [9]}, "tokens:"),
        ({"logits": [1.0, 2.0]}, "over tolerance"),
        ({"logits": [[0.1, 0.2]]}, "shape"),
        ({"logits": [float("nan")]}, "non-finite"),
        ({"checks": {"finite": False}}, "false"),
        ({"checks": {}}, "no named checks"),
        ({"updated": False}, "updated:"),
        ({"tokens": None}, "tokens: missing"),
    ],
)
def test_regressions_fail(fleet, capsys, change, expected):
    actual = {**fleet[3], **change}
    fleet[1].write_text(
        "import json\nprint(json.loads(" + repr(json.dumps(json.dumps(actual))) + "))\n"
    )
    code, rows = invoke(fleet, capsys)
    assert code == 1 and rows[0]["status"] == "fail"
    assert any(expected in failure for failure in rows[0]["failures"])


def test_missing_baseline_is_a_failure(fleet, capsys):
    fleet[2].unlink()
    code, rows = invoke(fleet, capsys)
    assert code == 1 and "no reviewed baseline" in rows[0]["failures"][0]


def test_exception_is_an_error(fleet, capsys):
    fleet[1].write_text("raise RuntimeError('broken contract')\n")
    code, rows = invoke(fleet, capsys)
    assert code == 1 and rows[0]["status"] == "error"
    assert "broken contract" in rows[0]["failures"][0]


def test_wrong_baseline_target_fails(fleet, capsys):
    baseline = json.loads(fleet[2].read_text())
    baseline["recorded_with"]["target"] = "another-target"
    fleet[2].write_text(json.dumps(baseline))
    code, rows = invoke(fleet, capsys)
    assert code == 1 and "baseline identity differs" in rows[0]["failures"][0]


def test_no_selection_is_success_even_with_explicit_target(fleet, capsys, monkeypatch):
    changed = fleet[0].parent / "changed.txt"
    changed.write_text("docs/index.md\n")
    monkeypatch.setattr(
        run, "fixture_folder", lambda *a: pytest.fail("unnecessary Hub download")
    )
    code, rows = invoke(
        fleet,
        capsys,
        "--changed-files",
        str(changed),
        "--contract",
        "demo",
        "--target",
        "demo-slim",
    )
    assert code == 0 and rows == []


def test_unknown_target_fails(fleet, capsys):
    code, rows = invoke(fleet, capsys, "--target", "unknown")
    assert code == 1 and rows == []


def test_selection_includes_processor_dependencies():
    dirs = {
        "llama": ["llama", "gpt2"],
        "smolvlm": ["smolvlm", "llama", "gpt2"],
        "bert": ["bert"],
    }
    assert set(
        select_contracts.select(
            ["src/transformers/models/gpt2/tokenization_gpt2.py"], dirs
        )
    ) == {"llama", "smolvlm"}
    assert set(
        select_contracts.select(["src/transformers/modeling_utils.py"], dirs)
    ) == set(dirs)
    assert select_contracts.select(["docs/index.md"], dirs) == {}


def test_otel_exports_case_failure_with_assertion(fleet, capsys, monkeypatch):
    memory = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setattr(instrument, "build_primary_exporter", lambda env: memory)
    monkeypatch.setattr(instrument, "build_staging_exporter", lambda env: None)
    fleet[1].write_text(
        "import json\nprint(json.dumps(" + repr({**fleet[3], "tokens": [9]}) + "))\n"
    )
    code, _ = invoke(fleet, capsys)
    assert code == 1
    spans = {s.name: s for s in memory.get_finished_spans()}
    case = spans["tests/contracts::demo::demo-slim[default]"]
    assert case.attributes["pytest.span_type"] == "test"
    assert case.attributes["pytest.nodeid"] == case.name
    assert case.attributes["mic.status"] == "fail"
    assert case.attributes["mic.variant"] == "default"
    assert case.status.status_code.name == "ERROR"
    assert case.end_time > case.start_time
    error = next(e for e in case.events if e.name == "exception")
    assert error.attributes["exception.type"] == "MICAssertionError"
    assert "tokens:" in error.attributes["exception.stacktrace"]
    assert "pytest.span_type" not in spans["fast-integration"].attributes


def test_legacy_fixture_download_uses_new_org(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/tmp/snapshot",
    )
    run.fixture_folder(
        Path("."), "hf-internal-testing/mic-bert-slim", {"revision": "immutable"}, False
    )
    assert calls == [("huggingface-mic/mic-bert-slim", {"revision": "immutable"})]


def test_umbrella_loads_expectations_alongside_usage(fleet, capsys):
    root = fleet[0].parent
    lock = json.loads(fleet[0].read_text())
    pin = lock["contracts"]["demo"]["fixtures"].pop("test/demo-slim")
    pin["target"] = "demo-slim"
    lock["contracts"]["demo"]["fixtures"]["test/demo-mic"] = pin
    fleet[0].write_text(json.dumps(lock))
    destination = (
        root / "integrations/transformers/expectations/cpu-test/demo-slim.json"
    )
    destination.parent.mkdir(parents=True)
    fleet[2].rename(destination)
    (root / ".mic-repo.json").write_text(
        json.dumps(
            {
                "format": "mic-umbrella-v1",
                "contract_files": ["integration.json", "usage.py"],
            }
        )
    )
    code, rows = invoke(fleet, capsys, "--target", "demo-slim")
    assert code == 0 and rows[0]["status"] == "pass"
    assert rows[0]["target"] == "demo-slim"
    assert "integrations/transformers/expectations" in rows[0]["baseline"]


def test_umbrella_hashes_keep_usage_weights_and_expectations_separate(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "integration.json").write_text("{}")
    usage = tmp_path / "integrations/transformers/usage.md"
    usage.parent.mkdir(parents=True)
    usage.write_text("usage")
    (tmp_path / ".mic-repo.json").write_text(
        json.dumps(
            {
                "format": "mic-umbrella-v1",
                "contract_files": [
                    "integration.json",
                    "integrations/transformers/usage.md",
                ],
            }
        )
    )
    before = run.contract_files(tmp_path)
    assert set(before) == {"integration.json", "integrations/transformers/usage.md"}
    assert set(run.file_hashes(tmp_path)) == {"config.json", "model.safetensors"}
    expectations = usage.parent / "expectations/cpu-test/target.json"
    expectations.parent.mkdir(parents=True)
    expectations.write_text("{}")
    assert run.contract_files(tmp_path) == before
    usage.write_text("changed usage")
    assert run.contract_files(tmp_path) != before


def test_selection_tracks_nested_model_configs(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    auto = ModuleType("transformers.models.auto")
    auto.configuration_auto = SimpleNamespace(
        CONFIG_MAPPING_NAMES={
            "musicgen_melody": "MusicgenMelodyConfig",
            "t5": "T5Config",
            "encodec": "EncodecConfig",
        }
    )
    auto.tokenization_auto = SimpleNamespace(TOKENIZER_MAPPING_NAMES={})
    auto.feature_extraction_auto = SimpleNamespace(FEATURE_EXTRACTOR_MAPPING_NAMES={})
    auto.image_processing_auto = SimpleNamespace(IMAGE_PROCESSOR_MAPPING_NAMES={})
    auto.processing_auto = SimpleNamespace(PROCESSOR_MAPPING_NAMES={})
    monkeypatch.setitem(sys.modules, "transformers.models.auto", auto)
    directories = {
        "MusicgenMelodyConfig": "musicgen_melody",
        "T5Config": "t5",
        "EncodecConfig": "encodec",
    }
    monkeypatch.setattr(select_contracts, "module_dir", directories.get)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "musicgen_melody",
                "text_encoder": {"model_type": "t5"},
                "audio_encoder": {"model_type": "encodec"},
            }
        )
    )
    assert select_contracts.code_dirs(tmp_path) == ["encodec", "musicgen_melody", "t5"]


def test_fleet_pin_changes_select_every_contract():
    cases = {"mic-bert": ["bert"], "mic-vit": ["vit"]}
    assert set(
        select_contracts.select(["tests/contracts/prototype-fleet.lock"], cases)
    ) == set(cases)


@pytest.mark.parametrize("explicit_target", [None, "mic-demo-slim"])
@pytest.mark.parametrize("unsupported", [False, True])
def test_fixture_target_identity(tmp_path, monkeypatch, explicit_target, unsupported):
    pin = {"revision": "b" * 40}
    if explicit_target:
        pin["target"] = explicit_target
    contract = {"fixtures": {"test/demo-mic": pin}}
    integration = {"targets": ["fixtures"]}
    if unsupported:
        integration = {"targets": [], "unsupported": {"fixtures": "unavailable"}}
    manifest = {
        "fixtures": {"test/demo-mic": {}},
        "integrations": {"transformers": integration},
    }
    monkeypatch.setattr(run, "fixture_folder", lambda *args: tmp_path)
    monkeypatch.setattr(run, "fixture_problem", lambda *args: None)
    rows = list(
        run.targets(tmp_path, contract, manifest, "transformers", fixtures_only=True)
    )
    assert len(rows) == 1
    assert rows[0][0] == (explicit_target or "demo-mic")
    assert rows[0][4] == ("unavailable" if unsupported else None)


def test_published_fixture_build_options_are_validated(fleet):
    manifest_path = fleet[0].parent / "integration.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fixtures"]["test/demo-slim"]["build"] = {
        "seed": 0,
        "config_overrides": {
            "fields": {"num_local_experts": 0},
            "reason": "Dense fixture",
        },
        "copy_assets": ["attention_mask.safetensors"],
        "preserve_frozen_prefixes": ["model.geometry."],
    }
    run.validate(manifest, "integration", manifest_path)
    del manifest["fixtures"]["test/demo-slim"]["build"]["config_overrides"]["reason"]
    with pytest.raises(SystemExit, match="reason"):
        run.validate(manifest, "integration", manifest_path)


@pytest.mark.parametrize("container", ["dict", "ordered"])
def test_registry_only_addition_selects_affected_models(container):
    path = "src/transformers/models/auto/configuration_auto.py"
    if container == "dict":
        before = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}\n'
        after = (
            'CONFIG_MAPPING_NAMES = {"new_model": "NewConfig", "bert": "BertConfig"}\n'
        )
    else:
        before = 'CONFIG_MAPPING_NAMES = OrderedDict([("bert", "BertConfig")])\n'
        after = 'CONFIG_MAPPING_NAMES = OrderedDict([("new_model", "NewConfig"), ("bert", "BertConfig")])\n'
    dirs = {"New": ["new_model"], "BERT": ["bert"], "Whisper": ["whisper"]}
    selected = select_contracts.select([path], dirs, {path: (before, after)})
    assert set(selected) == {"New"}
    assert (
        select_contracts.select([path], {"BERT": ["bert"]}, {path: (before, after)})
        == {}
    )


@pytest.mark.parametrize(
    "after",
    [
        'CONFIG_MAPPING_NAMES = {"bert": "ChangedConfig", "new": "NewConfig"}',
        'CONFIG_MAPPING_NAMES = {"new": "NewConfig"}',
        'CONFIG_MAPPING_NAMES = {"bert": "BertConfig", "new": "NewConfig"}\ndef shared(): return 2',
        'import new_dependency\nCONFIG_MAPPING_NAMES = {"bert": "BertConfig", "new": "NewConfig"}\ndef shared(): return 1',
        'CONFIG_MAPPING_NAMES = {"bert": "BertConfig", "new": "NewConfig", "new": "OtherConfig"}\ndef shared(): return 1',
        "not valid Python!",
    ],
)
def test_registry_changes_with_runtime_edits_remain_full_fleet(after):
    path = "src/transformers/models/auto/configuration_auto.py"
    before = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}\ndef shared(): return 1'
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == set(
        dirs
    )


@pytest.mark.parametrize(
    "path,module",
    [
        ("src/transformers/models/__init__.py", "new_model"),
        ("src/transformers/__init__.py", "models.new_model"),
    ],
)
def test_type_checking_model_registration_is_selective(path, module):
    before = "if TYPE_CHECKING:\n    from .models.bert import BertModel\n"
    after = before + f"    from .{module} import NewModel\n"
    dirs = {"New": ["new_model"], "BERT": ["bert"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == {
        "New"
    }


def test_unrelated_shared_lists_and_missing_diff_keep_full_fleet():
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    path = "src/transformers/generation/utils.py"
    assert set(
        select_contracts.select(
            [path],
            dirs,
            {
                path: (
                    'INPUTS = ["input_ids"]',
                    'INPUTS = ["input_ids", "padding_mask"]',
                )
            },
        )
    ) == set(dirs)
    assert set(
        select_contracts.select(["src/transformers/models/auto/modeling_auto.py"], dirs)
    ) == set(dirs)


@pytest.mark.parametrize(
    "value", ["register_and_mutate()", "(x := 1)", "[do_something() for _ in things]"]
)
def test_registry_entry_with_executable_side_effects_keeps_full_fleet(value):
    path = "src/transformers/models/auto/configuration_auto.py"
    before = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}'
    after = f'CONFIG_MAPPING_NAMES = {{"bert": "BertConfig", "new": {value}}}'
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == set(
        dirs
    )


def test_registry_model_alias_and_backend_guard_select_dependencies():
    path = "src/transformers/models/auto/tokenization_auto.py"
    before = 'TOKENIZER_MAPPING_NAMES = {"bert": "BertTokenizer"}'
    after = 'TOKENIZER_MAPPING_NAMES = {"bert": "BertTokenizer", "new_alias": "NewTokenizer" if is_tokenizers_available() else None}'
    dirs = {"New": ["shared_tokenizer"], "BERT": ["bert"]}
    assert set(
        select_contracts.select(
            [path],
            dirs,
            {path: (before, after)},
            model_type_to_module=lambda name: "shared_tokenizer",
        )
    ) == {"New"}


def test_changed_sources_uses_merge_base_and_falls_back_when_missing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init")
    git("config", "user.email", "mic-test@example.com")
    git("config", "user.name", "MIC test")
    path = "src/transformers/models/auto/configuration_auto.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    original = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}\n'
    file.write_text(original)
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    file.write_text(original.replace('"bert":', '"new": "NewConfig", "bert":'))
    git("add", ".")
    git("commit", "-m", "PR registration")
    head = git("rev-parse", "HEAD")
    git("checkout", "--detach", base)
    file.write_text(original + "# base branch advanced\n")
    git("add", ".")
    git("commit", "-m", "base advanced")
    advanced = git("rev-parse", "HEAD")
    git("checkout", "--detach", head)
    assert select_contracts.changed_sources([path], advanced) == {
        path: (original, file.read_text())
    }
    assert select_contracts.changed_sources([path], "missing-ref") == {}

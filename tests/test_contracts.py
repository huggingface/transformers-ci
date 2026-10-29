"""Exercise MIC execution, strict assertions, selection, and real OTEL export."""

import json
import subprocess
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from transformersci.contracts import fleet as fleet_api
from transformersci.contracts import report, run, select_contracts
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


@pytest.mark.parametrize("constructor", ["OrderedDict", "OrderedDict[str, str | None]"])
def test_registry_preserves_existing_duplicates_and_typed_constructor(constructor):
    path = "src/transformers/models/auto/modeling_auto.py"
    before = f'{constructor}([("sam3_tracker", "Sam3Tracker"), ("sam3_tracker", "Sam3Tracker")])'
    before = "MODEL_MAPPING_NAMES = " + before
    after = before.replace(
        '[("sam3_tracker"', '[("new", "NewModel"), ("sam3_tracker"', 1
    )
    dirs = {"New": ["new"], "SAM": ["sam3_tracker"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == {
        "New"
    }
    changed = after.replace(
        '("sam3_tracker", "Sam3Tracker")', '("sam3_tracker", "Changed")', 1
    )
    assert set(select_contracts.select([path], dirs, {path: (before, changed)})) == set(
        dirs
    )


def test_registry_additions_preserve_composed_mapping_splices():
    path = "src/transformers/models/auto/modeling_auto.py"
    before = "MODEL_MAPPING_NAMES = OrderedDict([*list(BASE_MAPPING_NAMES.items())])"
    after = before.replace("])", ', ("new", "NewModel")])')
    dirs = {"New": ["new"], "BERT": ["bert"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == {
        "New"
    }
    changed = after.replace("BASE_MAPPING_NAMES", "OTHER_MAPPING_NAMES")
    assert set(select_contracts.select([path], dirs, {path: (before, changed)})) == set(
        dirs
    )


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


def test_hub_fleet_metadata_is_in_results_traces_and_reproduction(
    fleet, capsys, monkeypatch, tmp_path
):
    revision = "d" * 40

    def lock_path(name, ref):
        assert (name, ref) == ("prototypes", revision)
        return fleet[0], revision

    monkeypatch.setattr(fleet_api, "lock_path", lock_path)
    memory = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setattr(instrument, "build_primary_exporter", lambda env: memory)
    monkeypatch.setattr(instrument, "build_staging_exporter", lambda env: None)
    assert (
        run.main(
            ["--fleet", "prototypes", "--fleet-revision", revision, "--fixtures-only"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out.splitlines()[0])
    assert result["fleet_repository"] == fleet_api.REPOSITORY
    assert result["fleet_revision"] == revision
    spans = [
        span
        for span in memory.get_finished_spans()
        if span.attributes.get("pytest.span_type") == "test"
    ]
    assert spans[0].attributes["mic.fleet_repository"] == fleet_api.REPOSITORY
    assert spans[0].attributes["mic.fleet_revision"] == revision
    result.update(status="fail", failures=["tokens differ"])
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n")
    report.main([str(path)])
    summary = capsys.readouterr().out
    assert f"--fleet prototypes --fleet-revision {revision}" in summary
    assert "--lock " not in summary


def test_custom_lock_cannot_silently_ignore_a_dataset_revision(fleet):
    with pytest.raises(SystemExit) as error:
        run.main(["--lock", str(fleet[0]), "--fleet-revision", "main"])
    assert error.value.code == 2


def test_framework_dataset_pin_is_strict_and_selects_the_entire_fleet(tmp_path):
    path = tmp_path / "contracts.lock"
    path.write_text("e" * 40 + "\n")
    assert fleet_api.read_revision(path) == "e" * 40
    for value in ("main", "", "e" * 40 + "\nother"):
        path.write_text(value)
        with pytest.raises(ValueError, match="immutable dataset"):
            fleet_api.read_revision(path)
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    assert set(select_contracts.select(["tests/contracts.lock"], dirs)) == set(dirs)


def test_runner_reads_framework_pin_by_default(fleet, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "tests/contracts.lock"
    path.parent.mkdir()
    path.write_text("f" * 40 + "\n")
    calls = []

    def lock_path(name, revision):
        calls.append((name, revision))
        return fleet[0], revision

    monkeypatch.setattr(fleet_api, "lock_path", lock_path)
    assert run.main(["--fixtures-only"]) == 0
    assert calls == [("control", "f" * 40)]


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
    assert run.load(path, "fleet-lock") == FLEET_LOCK


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

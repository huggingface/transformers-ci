import base64
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap


WORKFLOW = (
    Path(__file__).parents[1] / ".github" / "workflows" / "pr-ci-security-gate.yml"
)
STEP_NAME = "Materialize changed Python files from immutable blobs"


def _step_script(name: str) -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    step_marker = f"      - name: {name}\n"
    step_start = text.find(step_marker)
    assert step_start >= 0, f"workflow step not found: {name}"

    run_marker = "        run: |\n"
    run_start = text.find(run_marker, step_start)
    assert run_start >= 0, f"run block not found for workflow step: {name}"
    run_start += len(run_marker)

    boundaries = [text.find("\n      - name: ", run_start)]
    next_job = re.search(r"^  [A-Za-z0-9_-]+:\n", text[run_start:], re.MULTILINE)
    if next_job:
        boundaries.append(run_start + next_job.start())
    boundaries = [boundary for boundary in boundaries if boundary >= 0]
    return textwrap.dedent(text[run_start : min(boundaries, default=len(text))])


def _write_fake_gh(bin_dir: Path) -> None:
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

request = sys.argv[-1]
sha = request.rsplit("/", 1)[-1]
blobs = json.loads(os.environ["FAKE_GH_BLOBS"])
if sha not in blobs:
    print(f"unexpected blob request: {request}", file=sys.stderr)
    raise SystemExit(2)
print(json.dumps(blobs[sha]))
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)


def _run_materializer(
    tmp_path: Path,
    pr_files: list[dict[str, str]],
    python_files: list[str],
    blobs: dict[str, dict[str, object]],
) -> subprocess.CompletedProcess[str]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pr_files_path = inputs / "pr_files.json"
    py_files_path = inputs / "py_files.txt"
    pr_files_path.write_text(json.dumps(pr_files), encoding="utf-8")
    py_files_path.write_text(
        "".join(f"{name}\n" for name in python_files), encoding="utf-8"
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_gh(bin_dir)

    env = os.environ.copy()
    env.update(
        {
            "FAKE_GH_BLOBS": json.dumps(blobs),
            "GH_TOKEN": "test-token",
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PR_FILES_JSON": str(pr_files_path),
            "PR_SOURCE_DIR": str(tmp_path / "pr-source"),
            "PY_FILES_LIST": str(py_files_path),
            "REPO": "huggingface/transformers",
        }
    )
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step_script(STEP_NAME)],
        capture_output=True,
        text=True,
        env=env,
    )


def _blob(sha: str, content: bytes) -> dict[str, object]:
    return {
        "content": base64.b64encode(content).decode("ascii"),
        "encoding": "base64",
        "sha": sha,
        "size": len(content),
    }


def test_materializes_changed_python_files_from_immutable_blobs(tmp_path: Path) -> None:
    sha = "a" * 40
    content = b"print('immutable source')\n"

    result = _run_materializer(
        tmp_path,
        [{"filename": "src/example.py", "sha": sha, "status": "added"}],
        ["src/example.py"],
        {sha: _blob(sha, content)},
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pr-source" / "src" / "example.py").read_bytes() == content
    assert "Materialized 1 changed Python file(s)" in result.stdout


def test_rejects_python_paths_that_escape_the_materialization_root(
    tmp_path: Path,
) -> None:
    sha = "b" * 40
    filename = "src/../../escaped.py"

    result = _run_materializer(
        tmp_path,
        [{"filename": filename, "sha": sha, "status": "added"}],
        [filename],
        {sha: _blob(sha, b"print('escaped')\n")},
    )

    assert result.returncode != 0
    assert "Unsafe Python path" in result.stderr
    assert not (tmp_path / "escaped.py").exists()

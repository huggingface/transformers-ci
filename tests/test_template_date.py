"""Chat templates see the runner's fixed date, not today's."""

import datetime
import json
import os
import subprocess
import sys

import pytest

from transformersci.contracts import run as runner
from transformersci.contracts import template_date

RENDER = (
    "from transformers.utils.chat_template_utils import _compile_jinja_template\n"
    "import datetime, json\n"
    "print(json.dumps([_compile_jinja_template(\"{{ strftime_now('%Y-%m-%d') }}\").render(),"
    " datetime.datetime.now().year]))\n"
)
FREEZE = "from transformersci.contracts import template_date\ntemplate_date.freeze('2026-01-01')\n"


def fresh(code, env=None):
    # A fresh interpreter: freezing installs an import hook and patches a module,
    # both process-wide.
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def needs_transformers():
    try:
        import jinja2  # noqa: F401
        import transformers.utils.chat_template_utils  # noqa: F401
    except ImportError:
        pytest.skip("transformers with jinja2 is not installed")


@pytest.mark.parametrize("imported_first", [False, True])
def test_templates_see_the_frozen_date_whenever_transformers_is_imported(
    imported_first,
):
    needs_transformers()
    before = "import transformers.utils.chat_template_utils\n" if imported_first else ""
    rendered, year = fresh(before + FREEZE + RENDER)
    assert rendered == "2026-01-01"
    assert year == datetime.datetime.now().year  # only the template module's clock


def test_isolated_contract_processes_freeze_the_date():
    needs_transformers()
    env = {**os.environ, "MIC_TEMPLATE_DATE": "2026-01-01"}
    rendered, _ = fresh(runner.patch_prelude([]) + RENDER, env)
    assert rendered == "2026-01-01"


def test_freezing_does_not_import_transformers():
    code = (
        FREEZE
        + "import json, sys\nprint(json.dumps(['transformers' in sys.modules]))\n"
    )
    assert fresh(code) == [False]


def test_a_malformed_date_is_refused():
    with pytest.raises(ValueError):
        template_date.freeze("January 1st")


def test_the_baseline_schema_accepts_the_template_date():
    schema = json.loads(
        (runner.SCHEMAS / "baseline.schema.json").read_text(encoding="utf-8")
    )
    assert "template_date" in schema["properties"]["recorded_with"]["properties"]
    datetime.date.fromisoformat(template_date.TEMPLATE_DATE)

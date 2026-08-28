# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dashboard links a Serge task PR body should carry — the ``test_links`` field
of ``POST /tasks``.

Serge renders these links but deliberately knows nothing about them: it holds no
Grafana URL, no dashboard UID and no template-variable names, because it serves
repos whose CI is not observed here. **This** is the side that owns the URL
scheme, next to the ``dashboard/*.json`` that defines those variables — so a UID
or variable rename never needs a Serge release to follow it.

The payload is a map of pytest node-id → list of ``{label, url}``, keyed by
node-id because one task can carry several candidate failure groups and Serge
renders only the links of the group its patch actually fixed.
"""

from __future__ import annotations

from urllib.parse import urlencode

# Per-test view of the transformers-ci Grafana stack. `test_nodeid` is a textbox
# variable, so the node-id alone drives the summary, the duration history *and*
# the traceback panel (which falls back to the newest `trace_id` recorded for the
# node-id when the link carries none). Callers that already know the trace should
# still pass it — it pins the panel to that exact run.
TEST_DASHBOARD = "/d/pytest-test/test"

# The daily integration failures dispatched to Serge all come from this job on
# main; pinning both keeps the dashboard's own variables consistent.
DAILY_JOB = "run_models_gpu"
DAILY_PR = "main"

# The dashboard defaults to `now-24h`, but its failure metadata (exception type,
# traceback) is scraped per run: a PR body outlives that window and the link would
# then open on an empty panel. The triage only dispatches tests failing on most of
# the last 7 days, so a 7d window is the one that matches the data behind it.
WINDOW_FROM = "now-7d"
WINDOW_TO = "now"

_LABEL = "Test dashboard"


def grafana_test_url(
    base_url: str,
    node_id: str,
    *,
    test_job: str = DAILY_JOB,
    pr: str = DAILY_PR,
    trace_id: str = "",
    run_id: str = "",
) -> str:
    """Deep link to the per-test view for ``node_id``.

    ``""`` when either the base URL or the node-id is missing, so an unconfigured
    Grafana simply produces no links rather than a broken one. ``trace_id`` /
    ``run_id`` are optional: pass them when the caller knows them (they pin the
    traceback and run panels to that run), omit them to let the dashboard resolve
    the newest failure for the node-id itself.

    The time range is pinned to :data:`WINDOW_FROM` so the link still resolves
    days after the PR body was written.
    """
    if not base_url or not node_id:
        return ""
    params = {"var-test_nodeid": node_id}
    if test_job:
        params["var-test_job"] = test_job
    if pr:
        params["var-pr"] = pr
    if trace_id:
        params["var-trace_id"] = trace_id
    if run_id:
        params["var-run_id"] = run_id
    params["from"] = WINDOW_FROM
    params["to"] = WINDOW_TO
    return f"{base_url.rstrip('/')}{TEST_DASHBOARD}?{urlencode(params)}"


def test_links(
    base_url: str,
    node_ids: list[str],
    *,
    label: str = _LABEL,
    test_job: str = DAILY_JOB,
    pr: str = DAILY_PR,
    trace_ids: dict[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """The ``test_links`` payload for a set of failing tests.

    Duplicate node-ids collapse; anything that yields no URL is dropped, so an
    unset ``base_url`` returns ``{}`` and the task is dispatched without the
    field (Serge then renders no link section — same as before this existed).
    """
    out: dict[str, list[dict[str, str]]] = {}
    for node_id in node_ids:
        if not node_id or node_id in out:
            continue
        url = grafana_test_url(
            base_url,
            node_id,
            test_job=test_job,
            pr=pr,
            trace_id=(trace_ids or {}).get(node_id, ""),
        )
        if url:
            out[node_id] = [{"label": label, "url": url}]
    return out

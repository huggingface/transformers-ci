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
"""Render the stored state as ``ci_github_*`` Prometheus series.

Separate from the trace-derived ``pytest_*`` metrics on purpose: these say what
GitHub says about execution, never what tests found. The shape keeps series
from going stale:

* ``ci_github_{run,job}_status`` is a NUMBER on fixed identity labels
  (1 queued, 2 in progress, 3 completed), so a transition updates one series
  instead of leaving the old state's series behind;
* the conclusion, known only once completed and then final, is the one state
  carried as a label (``ci_github_{run,job}_conclusion_info``);
* times are separate ``*_timestamp_seconds`` gauges.

``run_id`` is ``"<id>:<attempt>"`` and ``pr`` follows the exporter (a PR number,
else the branch of a push/schedule run), so these series join the ``pytest_*``
ones on the same labels. Titles, logs and error text are never labels. A
completed record is published for ``completed_window_seconds`` after it
finished; Prometheus keeps the history, the payload stays small.
"""

from __future__ import annotations

from .reducer import COMPLETED, status_rank

# Events whose runs the exporter files under their branch rather than a PR.
_BRANCH_EVENTS = frozenset({"push", "schedule", "workflow_dispatch"})


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(labels: dict[str, str]) -> str:
    return (
        "{"
        + ",".join(f'{key}="{_escape(value)}"' for key, value in labels.items())
        + "}"
    )


def run_label(state: dict) -> str:
    return f"{state['run_id']}:{state['attempt']}"


def pr_label(state: dict) -> str:
    """The PR a run belongs to; its branch for branch runs; "" when unknown
    (a fork PR before enrichment) — never a guess."""
    prs = state.get("prs") or ()
    if prs:
        return str(min(prs))
    if state.get("event") in _BRANCH_EVENTS:
        return str(state.get("head_branch") or "")
    return ""


class _Family:
    def __init__(self, name: str, kind: str, help_text: str) -> None:
        self.name, self.kind, self.help_text = name, kind, help_text
        self.samples: list[str] = []

    def add(self, labels: dict[str, str], value: float | int) -> None:
        if isinstance(value, float):
            text = f"{value:.3f}"
        else:
            text = str(value)
        self.samples.append(f"{self.name}{_labels(labels)} {text}")

    def lines(self) -> list[str]:
        if not self.samples:
            return []
        return [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.kind}",
            *self.samples,
        ]


def render(
    runs: list[dict],
    jobs: list[dict],
    *,
    service: dict[str, float | int | dict[str, int]],
) -> str:
    """The ``/metrics`` payload for ``runs``/``jobs`` plus service health."""
    families = {
        name: _Family(name, "gauge", help_text)
        for name, help_text in (
            (
                "ci_github_run_info",
                "A workflow run attempt GitHub reported (identity and grouping).",
            ),
            (
                "ci_github_run_status",
                "Run status from GitHub: 1 queued, 2 in progress, 3 completed.",
            ),
            (
                "ci_github_run_conclusion_info",
                "Conclusion of a completed run, verbatim from GitHub.",
            ),
            (
                "ci_github_run_created_timestamp_seconds",
                "When GitHub created the run attempt.",
            ),
            (
                "ci_github_run_started_timestamp_seconds",
                "When the run attempt started.",
            ),
            (
                "ci_github_run_completed_timestamp_seconds",
                "When the run attempt completed.",
            ),
            (
                "ci_github_run_needs_lookup",
                "1 when two deliveries disagreed on the run's conclusion.",
            ),
            (
                "ci_github_job_info",
                "A job of a run attempt (identity and display name).",
            ),
            (
                "ci_github_job_status",
                "Job status from GitHub: 1 queued, 2 in progress, 3 completed.",
            ),
            (
                "ci_github_job_conclusion_info",
                "Conclusion of a completed job, verbatim from GitHub.",
            ),
            (
                "ci_github_job_created_timestamp_seconds",
                "When GitHub created (queued) the job.",
            ),
            (
                "ci_github_job_started_timestamp_seconds",
                "When a runner started the job.",
            ),
            ("ci_github_job_completed_timestamp_seconds", "When the job completed."),
            (
                "ci_github_job_needs_lookup",
                "1 when two deliveries disagreed on the job's conclusion.",
            ),
        )
    }
    run_prs: dict[tuple[str, str], str] = {}
    for run in runs:
        identity = {"repository": run["repository"], "run_id": run_label(run)}
        pr = pr_label(run)
        run_prs[(run["repository"], identity["run_id"])] = pr
        families["ci_github_run_info"].add(
            {
                **identity,
                "workflow": run.get("workflow") or "",
                "event": run.get("event") or "",
                "pr": pr,
            },
            1,
        )
        _progress(families, "run", identity, run, run.get("updated_at"))

    for job in jobs:
        run_id = f"{job['run_id']}:{job['attempt']}"
        identity = {
            "repository": job["repository"],
            "run_id": run_id,
            "job_id": str(job["job_id"]),
        }
        families["ci_github_job_info"].add(
            {
                **identity,
                "pr": run_prs.get((job["repository"], run_id), ""),
                "name": job.get("name") or "",
            },
            1,
        )
        _progress(families, "job", identity, job, job.get("completed_at"))

    lines: list[str] = []
    for family in families.values():
        lines.extend(family.lines())
    lines.extend(_service_lines(service))
    return "\n".join(lines) + "\n"


def _progress(
    families: dict[str, _Family],
    kind: str,
    identity: dict[str, str],
    state: dict,
    completed_at: float | None,
) -> None:
    rank = status_rank(state.get("status") or "")
    if rank:
        families[f"ci_github_{kind}_status"].add(identity, rank)
    if state.get("status") == COMPLETED:
        families[f"ci_github_{kind}_conclusion_info"].add(
            {**identity, "conclusion": state.get("conclusion") or "unknown"}, 1
        )
        if completed_at:
            families[f"ci_github_{kind}_completed_timestamp_seconds"].add(
                identity, float(completed_at)
            )
    for field in ("created_at", "started_at"):
        if state.get(field):
            name = field.removesuffix("_at")
            families[f"ci_github_{kind}_{name}_timestamp_seconds"].add(
                identity, float(state[field])
            )
    if state.get("needs_lookup"):
        families[f"ci_github_{kind}_needs_lookup"].add(identity, 1)


def _service_lines(service: dict[str, float | int | dict[str, int]]) -> list[str]:
    deliveries = service.get("deliveries") or {}
    assert isinstance(deliveries, dict)
    lines = [
        "# HELP ci_github_status_deliveries_total Webhook deliveries by outcome since start.",
        "# TYPE ci_github_status_deliveries_total counter",
    ]
    for outcome in ("accepted", "duplicate", "ignored", "rejected", "error"):
        lines.append(
            f'ci_github_status_deliveries_total{{outcome="{outcome}"}} {int(deliveries.get(outcome, 0))}'
        )
    for name, kind, help_text in (
        (
            "processing_seconds_total",
            "counter",
            "Seconds spent verifying, parsing and committing deliveries.",
        ),
        (
            "last_delivery_timestamp_seconds",
            "gauge",
            "When the last accepted delivery was committed (0 = none since start).",
        ),
        ("runs_active", "gauge", "Stored runs not yet completed."),
        ("jobs_active", "gauge", "Stored jobs not yet completed."),
        ("runs_stored", "gauge", "Stored runs, active and completed."),
        ("jobs_stored", "gauge", "Stored jobs, active and completed."),
        (
            "needs_lookup",
            "gauge",
            "Stored runs/jobs whose conclusion deliveries disagreed on.",
        ),
        ("publication_timestamp_seconds", "gauge", "When this payload was rendered."),
    ):
        value = service.get(name, 0)
        lines.append(f"# HELP ci_github_status_{name} {help_text}")
        lines.append(f"# TYPE ci_github_status_{name} {kind}")
        lines.append(
            f"ci_github_status_{name} {value:.3f}"
            if isinstance(value, float)
            else f"ci_github_status_{name} {value}"
        )
    return lines

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
"""What a CI runner is: its type, and the hardware a job on it actually sees.

A self-hosted runner's name is its scale set plus a region/pool/pod suffix,
``aws-g5-4xlarge-cache-use1-public-80-x9tqf-runner-7bthr``. The scale set
(``aws-g5-4xlarge-cache``) is GitHub's ``runner_group_name`` and what the
dashboards call the *runner type*: a handful of values, unlike the name, which
is unique per job and never becomes a Prometheus label.

:func:`probe_hardware` describes the machine from inside the job, because that
is what a test gets: an ``aws-g5-12xlarge-cache`` job sees 2 of the instance's 4
GPUs. It runs in ``configure-ci-otel`` on every traced CI job, so it only reads
``/proc`` and cgroup files and asks ``nvidia-smi`` with a short timeout, and any
failure just leaves that fact out. Stdlib only: ci-github-status imports this.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping

# <scale set>-<region>-<pool>-<n>-<id>-runner-<id>
_RUNNER_NAME_RE = re.compile(
    r"^(?P<type>.+?)-[a-z]+\d+-[a-z]+-\d+-[a-z0-9]+-runner-[a-z0-9]+$"
)

# GitHub-hosted runners report this group (and names like "GitHub Actions 12").
GITHUB_HOSTED = "github-hosted"


def runner_type(runner_name: str | None, group: str | None = None) -> str:
    """The runner's type: GitHub's group when it has one, else the scale set
    parsed off the name; "" when neither says (e.g. a job not started yet)."""
    group = (group or "").strip()
    name = (runner_name or "").strip()
    if group == "GitHub Actions" or name.startswith("GitHub Actions"):
        return GITHUB_HOSTED
    if group and group != "Default":
        return group
    match = _RUNNER_NAME_RE.match(name)
    return match.group("type") if match else ""


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _cpu_model(read: Callable[[str], str]) -> str:
    for line in read("/proc/cpuinfo").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name" and value.strip():
            return " ".join(value.split())
    return ""


def _vcpus(read: Callable[[str], str]) -> str:
    # A container's CPU quota (cgroup v2), else the CPUs this process may use.
    try:
        quota, period = read("/sys/fs/cgroup/cpu.max").split()[:2]
        if quota != "max" and int(period) > 0:
            return f"{int(quota) / int(period):g}"
    except (OSError, ValueError):
        pass
    try:
        return str(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        count = os.cpu_count()
        return str(count) if count else ""


def _memory_gib(read: Callable[[str], str]) -> str:
    # A container's memory limit (cgroup v2), else the machine's.
    try:
        limit = read("/sys/fs/cgroup/memory.max").strip()
        if limit != "max":
            return f"{int(limit) / 2**30:.0f}"
    except (OSError, ValueError):
        pass
    for line in read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            return f"{int(line.split()[1]) / 2**20:.0f}"
    return ""


def _nvidia_gpus(run: Callable[[list[str]], str]) -> dict[str, str]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [
        [part.strip() for part in line.split(",")]
        for line in out.splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if len(row) == 2 and row[0]]
    if not rows:
        return {}
    names = sorted({row[0] for row in rows})
    return {
        "gpu_vendor": "nvidia",
        "gpu_model": " / ".join(names),
        "gpu_count": str(len(rows)),
        # nvidia-smi reports MiB.
        "gpu_memory_gib": f"{int(rows[0][1]) / 1024:.0f}",
    }


def _run(command: list[str]) -> str:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=5, check=True
    ).stdout


def probe_hardware(
    *,
    read: Callable[[str], str] = _read,
    run: Callable[[list[str]], str] = _run,
) -> dict[str, str]:
    """Facts about this machine as the job sees it; a fact that cannot be read
    is left out rather than guessed."""
    facts: dict[str, str] = {}
    for key, probe in (
        ("cpu_model", _cpu_model),
        ("vcpus", _vcpus),
        ("memory_gib", _memory_gib),
    ):
        try:
            value = probe(read)
        except (OSError, ValueError):
            continue
        if value:
            facts[key] = value
    try:
        facts.update(_nvidia_gpus(run))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return facts


# Resource-attribute prefix for the runner facts (transformers.test.runner.*).
ATTRIBUTE_PREFIX = "transformers.test.runner."


def runner_resource_attributes(
    env: Mapping[str, str],
    *,
    probe: Callable[[], dict[str, str]] = probe_hardware,
) -> list[str]:
    """``key=value`` resource attributes naming this job's runner and its
    hardware. Never raises: a CI job must not fail over a description."""
    name = env.get("RUNNER_NAME", "")
    if not name:
        # Not on a GitHub runner (a local run, a test): nothing to describe.
        return []
    try:
        facts = {"name": name, "type": runner_type(name)}
        facts.update(probe())
    except Exception:  # noqa: BLE001 - see docstring
        return []
    # OTEL_RESOURCE_ATTRIBUTES is comma-separated key=value pairs, and the SDK
    # percent-decodes values.
    return [
        f"{ATTRIBUTE_PREFIX}{key}={re.sub(r'[,=%]', ' ', value)}"
        for key, value in facts.items()
        if value
    ]


# Runner types no traced job runs on (the AMD pools run a workflow outside this
# repo; GitHub's own runners run only untraced steps), described by hand from
# their own job logs (``rocminfo`` in the "Check
# Runners" step). ``source`` names the log. The CPU is the host's: these logs do
# not show the job's CPU or memory share. mi250 is left out: no recent log.
DOCUMENTED_RUNNERS: dict[str, dict[str, str]] = {
    # GitHub's own runners run no traced job here; GitHub publishes their size.
    GITHUB_HOSTED: {
        "source": "GitHub docs: standard Linux runner, public repositories",
        "vcpus": "4",
        "memory_gib": "16",
    },
    # NVIDIA pools (daily CI, run-slow): nvidia-smi in their own job logs. They
    # describe themselves as measured once a job runs the fact sheet; until
    # then these keep them listed. The logs show no CPU or memory.
    "aws-g5-4xlarge-cache": {
        "source": "job log 2026-09-28 (job 108767465223)",
        "gpu_vendor": "nvidia",
        "gpu_model": "NVIDIA A10G",
        "gpu_count": "1",
        "gpu_memory_gib": "22",
    },
    "aws-g5-12xlarge-cache": {
        "source": "job log 2026-09-28 (job 108767465264): 2 of the instance's 4 GPUs",
        "gpu_vendor": "nvidia",
        "gpu_model": "NVIDIA A10G",
        "gpu_count": "2",
        "gpu_memory_gib": "22",
    },
    # Report/notification jobs; its log prints no hardware.
    "aws-general-8-plus": {
        "source": "job log 2026-09-27 (job 108558713975): no hardware printed",
    },
    "amd-mi300-1gpu": {
        "source": "job log 2026-09-25 (job 107956364031)",
        "cpu_model": "2x AMD EPYC 9654 96-Core (host)",
        "gpu_vendor": "amd",
        "gpu_model": "gfx942 (MI300 series)",
        "gpu_count": "1",
    },
    "amd-mi300-2gpu": {
        "source": "job log 2026-09-25 (job 107956363937)",
        "cpu_model": "2x AMD EPYC 9654 96-Core (host)",
        "gpu_vendor": "amd",
        "gpu_model": "gfx942 (MI300 series)",
        "gpu_count": "2",
    },
    "hfc-amd-mi355-1gpu": {
        "source": "job log 2026-09-27 (job 108558269803)",
        "cpu_model": "2x Intel(R) Xeon(R) Platinum 8480C (host)",
        "gpu_vendor": "amd",
        "gpu_model": "gfx950 (MI355 series)",
        "gpu_count": "1",
    },
    "hfc-amd-mi355-2gpu": {
        "source": "job log 2026-09-27 (job 108558269901)",
        "cpu_model": "2x Intel(R) Xeon(R) Platinum 8480C (host)",
        "gpu_vendor": "amd",
        "gpu_model": "gfx950 (MI355 series)",
        "gpu_count": "2",
    },
}

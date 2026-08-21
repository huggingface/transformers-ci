#!/usr/bin/env python3
"""Deploy the transformers-ci Helm chart.

Beyond running `helm upgrade`, this works out which pods a change actually needs
to restart, restarts only those, and then proves the stack came back healthy.

Why that matters here: Helm only rolls a pod when its *pod template* changes. A
ConfigMap-only change -- the common case for this chart, since Tempo, Grafana and
Prometheus all read their config from mounted ConfigMaps -- rewrites the
ConfigMap and leaves the running process on its old config. Only some workloads
in this chart carry a `checksum/*` pod annotation (otelcol, grafana's alerting,
prometheus' recording rules), so for the rest `helm upgrade` reports success
while the change sits inert until something restarts the pod.

The plan is computed by diffing the rendered manifest against the live release,
mapping each changed ConfigMap/Secret to the workloads that mount or reference
it, and then dropping anything Helm is going to roll anyway.

Note on secrets: credentials are delivered by a secret-sync CR, so a rotated
*value* is invisible to a manifest diff. Those are detected separately, by
comparing when the live Secret was last written against when each consuming pod
started -- see `find_stale_secret_consumers`. Only Secret *metadata* is read;
values are never fetched. See deploy/README.md.

Two further things a bare resource-level diff does not tell you, both of which
have shipped silently before:

* **What a change removes.** "change ConfigMap/grafana-dashboards" is equally
  true of adding a panel and of deleting a dashboard somebody applied by hand.
  The plan therefore reports added/removed line counts per resource, and calls
  out any config key that is live but absent from the render as a deletion --
  the shape of an accidental revert. `--diff` prints the actual hunks.

* **Which code the cloned workloads will run.** `traceExporter.sourceRevision`
  and friends are checked out by an init container at pod start, so the values
  file decides which commit prod runs, and a stale pin silently rolls the code
  *backwards* while `helm upgrade` reports success. The plan resolves each
  revision against the local clone and refuses a non-fast-forward move unless
  `--allow-source-rollback` says otherwise; `verify` then reads the revision
  back out of the running pod rather than trusting the annotation.
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CHART_DIR = ROOT_DIR / "deploy" / "helm"
DASHBOARD_DIR = CHART_DIR / "dashboards"

WORKLOAD_KINDS = ("Deployment", "StatefulSet", "DaemonSet")
CONFIG_KINDS = ("ConfigMap", "Secret")

# How a changed ConfigMap reaches the process that is already running.
# Anything unlisted defaults to RESTART: "config is read once at startup" is the
# common case, and an unnecessary restart only costs a few seconds, whereas a
# skipped one ships a deploy that silently does nothing.
APPLY_AUTO = "auto"
APPLY_RELOAD = "reload"
APPLY_RESTART = "restart"

CONFIG_APPLY = {
    # Grafana's file provisioner rescans the dashboard directory on its own poll
    # interval (10s unless updateIntervalSeconds says otherwise), so dashboard
    # JSON lands by itself once kubelet syncs the ConfigMap volume.
    "grafana-dashboards": APPLY_AUTO,
    "grafana-dashboards-restricted": APPLY_AUTO,
    # Prometheus runs with --web.enable-lifecycle, so it can re-read the config
    # in place; that keeps the TSDB head and avoids a scrape gap.
    "prometheus-config": APPLY_RELOAD,
}

# In-pod reload command per workload, used when CONFIG_APPLY says APPLY_RELOAD.
# Falls back to a restart when the workload has no entry here.
RELOAD_COMMANDS = {
    ("StatefulSet", "prometheus"): (
        "prometheus",
        ["wget", "-q", "-O-", "--post-data=", "http://127.0.0.1:9090/-/reload"],
    ),
}

# Restart order, dependencies before the things that consume them. The data path
# is otelcol -> tempo -> trace-exporter -> prometheus -> grafana, so the trace
# store goes first and the UI that queries everything goes last. Each workload is
# waited on before the next one starts, which also keeps tempo and otelcol from
# ever rolling at the same time -- during a tempo restart a healthy otelcol
# buffers and retries its OTLP exports, so spans are delayed rather than dropped.
# Workloads not listed here are restarted last, in name order.
RESTART_ORDER = (
    "tempo",
    "trace-exporter",
    "otelcol",
    "backup-status-exporter",
    "prometheus",
    "grafana",
)

# Pod-template annotation carrying the git revision an init container clones at
# pod start. The chart stamps it from `<workload>.sourceRevision` so that changing
# the pinned commit changes the pod template, which is what makes Helm roll the
# pod -- without it a new revision would sit in the values file with nothing to
# apply it.
SOURCE_REVISION_ANNOTATION = "rollout/source-revision"
# Init container that performs that clone. Named so the mount path can be derived
# from the manifest instead of hardcoded here.
CLONE_INIT_CONTAINER = "clone-src"

BAD_WAITING_REASONS = (
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "InvalidImageName",
)


def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing required command: {name}")


def run(
    args: list[str], *, check: bool = True, quiet: bool = False
) -> subprocess.CompletedProcess[str]:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.run(args, check=check, text=True, stdout=stdout, stderr=stderr)


def output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def try_output(args: list[str]) -> str | None:
    """Run a command, returning None instead of raising when it fails."""
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        return None


def note(message: str) -> None:
    """Print to stderr, keeping stdout clean for piped manifest renders."""
    print(message, file=sys.stderr)


# --------------------------------------------------------------------------
# manifest parsing
# --------------------------------------------------------------------------


def parse_json_stream(text: str) -> list[dict]:
    """Parse the concatenated JSON objects that `kubectl -o json` emits."""
    decoder = json.JSONDecoder()
    docs: list[dict] = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return docs
        doc, index = decoder.raw_decode(text, index)
        if isinstance(doc, dict) and doc.get("kind") == "List":
            docs.extend(doc.get("items") or [])
        elif isinstance(doc, dict):
            docs.append(doc)


def manifest_to_docs(manifest: str) -> list[dict]:
    """Convert a YAML manifest to dicts.

    Shelling out to kubectl keeps this script dependency-free (the sibling
    scripts are all stdlib-only, so a checkout needs no venv to deploy).
    """
    if not manifest.strip():
        return []
    proc = subprocess.run(
        ["kubectl", "create", "-o", "json", "--dry-run=client", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"failed to parse manifest:\n{proc.stderr.strip()}")
    return parse_json_stream(proc.stdout)


# --------------------------------------------------------------------------
# chart integrity (dashboard symlinks)
# --------------------------------------------------------------------------


def unresolved_symlink_target(path: Path) -> str | None:
    """Return the target path if `path` is a symlink stored as a plain text file.

    Git for Windows defaults to core.symlinks=false, which materialises a mode
    120000 entry as a small regular file whose entire contents are the link
    target. Such a file is one short line holding a path and nothing else.
    """
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
            return None
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text or "\n" in text or text.startswith("{"):
        return None
    if "/" not in text and "\\" not in text:
        return None
    return text


def check_chart_files() -> None:
    """Fail before deploying if the dashboard symlinks did not survive checkout.

    deploy/helm/dashboards/* are symlinks into dashboard/ (git mode 120000) and
    the chart reads them with .Files.Get. When a checkout materialises them as
    text files instead, Helm ships the *link target string* as the dashboard
    body, so Grafana silently serves "../../../dashboard/foo.json" in place of a
    dashboard. That only shows up by eyeballing the UI, so catch it here.
    """
    if not DASHBOARD_DIR.is_dir():
        return

    broken = []
    dangling = []
    for path in sorted(DASHBOARD_DIR.iterdir()):
        if path.is_dir():
            continue
        if path.is_symlink() and not path.exists():
            dangling.append(path.name)
            continue
        target = unresolved_symlink_target(path)
        if target is not None:
            broken.append((path.name, target))

    if dangling:
        raise SystemExit(
            "dangling symlinks in deploy/helm/dashboards: "
            + ", ".join(dangling)
            + "\nthe dashboard/ sources are missing from this checkout."
        )
    if not broken:
        return

    names = "\n".join(f"  {name} -> {target}" for name, target in broken)
    raise SystemExit(
        "refusing to deploy: deploy/helm/dashboards entries are unresolved symlinks.\n"
        f"{names}\n"
        "\nThese are git symlinks (mode 120000) into dashboard/, but this checkout\n"
        "has them as plain text files containing the link target -- what Git for\n"
        "Windows produces when core.symlinks is false (its default). Helm reads\n"
        "them with .Files.Get, so deploying would hand Grafana those path strings\n"
        "instead of dashboards.\n"
        "\nSymlinks do work on Windows, but need both an OS privilege and a git\n"
        "setting, and the setting only takes effect during a checkout:\n"
        "\n  1. Give Windows permission to create symlinks -- enable Developer Mode\n"
        "     (Settings > Privacy & security > For developers), or run git from an\n"
        "     elevated shell. Without this, git falls back to text files even when\n"
        "     core.symlinks is true.\n"
        "\n  2. Set core.symlinks when you clone (this is the fix that prevents it):\n"
        "       git clone -c core.symlinks=true <repo-url>\n"
        "     or make it the default for future clones:\n"
        "       git config --global core.symlinks true\n"
        "     (the Git for Windows installer's 'Enable symbolic links' option sets\n"
        "     the same thing)\n"
        "\n  3. To repair THIS checkout -- the setting does not convert files already\n"
        "     on disk, so delete them and let git restore them from the index:\n"
        "       git config core.symlinks true\n"
        "       Remove-Item deploy\\helm\\dashboards\\*     # PowerShell\n"
        "       git checkout -- deploy/helm/dashboards\n"
        "\nTo verify, check the working tree -- `git ls-files -s` is NOT a valid\n"
        "check here, because the index reports mode 120000 even when the working\n"
        "tree holds plain text files:\n"
        "       test -L deploy/helm/dashboards/<a-dashboard>.json"
    )


def validate_rendered_dashboards(docs: list[dict]) -> None:
    """Every dashboard shipped in a ConfigMap must be parseable JSON.

    This is the authoritative check behind check_chart_files: whatever mangles a
    dashboard on the way in -- an unresolved symlink, a truncated file, a bad
    merge -- it stops being valid JSON, and Grafana would accept the ConfigMap
    and quietly fail to render the dashboard.
    """
    problems = []
    for doc in docs:
        if doc.get("kind") != "ConfigMap":
            continue
        name = (doc.get("metadata") or {}).get("name") or ""
        if not name.startswith("grafana-dashboards"):
            continue
        for key, body in sorted((doc.get("data") or {}).items()):
            if not key.endswith(".json"):
                continue
            try:
                json.loads(body)
            except ValueError as exc:
                snippet = " ".join(body.split())[:80]
                problems.append(f"  {name}/{key}: {exc}\n    starts with: {snippet!r}")
    if problems:
        raise SystemExit(
            "refusing to deploy: rendered dashboard JSON is invalid.\n"
            + "\n".join(problems)
        )


# --------------------------------------------------------------------------
# rotated secrets
# --------------------------------------------------------------------------


def parse_k8s_time(value: str) -> datetime | None:
    """Parse an API-server RFC3339 timestamp (always UTC, sometimes fractional)."""
    value = value.strip()
    if not value.endswith("Z"):
        return None
    value = value[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def secret_write_times(namespace: str, names) -> dict:
    """Last write time per Secret, from managedFields.

    Reads **metadata only** -- the jsonpath never touches `.data`, so no secret
    value is fetched, logged or held in memory.

    A secret-sync operator writes the Secret only when the upstream value really
    changes (a no-op reconcile does not bump managedFields), and `helm upgrade`
    does not touch these Secrets at all, so this timestamp tracks rotations rather
    than deploy churn.
    """
    times = {}
    for name in sorted(names):
        raw = try_output(
            [
                "kubectl",
                "get",
                "secret",
                name,
                "-n",
                namespace,
                "--show-managed-fields=true",
                "-o",
                "jsonpath={.metadata.managedFields[*].time}",
            ]
        )
        if not raw:
            continue
        stamps = [parse_k8s_time(part) for part in raw.split()]
        stamps = [s for s in stamps if s]
        if stamps:
            times[name] = max(stamps)
    return times


def workload_pod_starts(doc: dict, namespace: str) -> list:
    """Container start times for a workload's currently running pods."""
    selector = ((doc.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}
    if not selector:
        return []
    label_selector = ",".join(f"{k}={v}" for k, v in sorted(selector.items()))
    raw = try_output(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            label_selector,
            "-o",
            "json",
        ]
    )
    if not raw:
        return []
    starts = []
    for pod in json.loads(raw).get("items") or []:
        if (pod.get("status") or {}).get("phase") != "Running":
            continue
        for status in (pod.get("status") or {}).get("containerStatuses") or []:
            started = ((status.get("state") or {}).get("running") or {}).get(
                "startedAt"
            )
            stamp = parse_k8s_time(started or "")
            if stamp:
                starts.append(stamp)
    return starts


def find_stale_secret_consumers(plan: Plan, new_by_key: dict, namespace: str) -> dict:
    """Workloads whose pods started before the Secret they consume was last written.

    This is the rotation case a manifest diff cannot see: the chart renders
    byte-identically before and after a rotation because the values live in the
    secret store, not the chart. Secrets are injected with `env … secretKeyRef`
    and read once at pod start, so a pod older than the Secret is serving the old
    credential.
    """
    consumed: dict = {}
    for key, doc in new_by_key.items():
        if key[0] not in WORKLOAD_KINDS:
            continue  # CronJobs re-read the Secret on their next run
        template = pod_template(doc)
        if template is None:
            continue
        _, secrets = config_refs(template)
        for name in secrets:
            consumed.setdefault(name, []).append(key)
    if not consumed:
        return {}

    written = secret_write_times(namespace, set(consumed))
    rolled = set(plan.rolled_by_helm)
    stale: dict = {}
    for name, targets in consumed.items():
        changed_at = written.get(name)
        if changed_at is None:
            continue
        for target in targets:
            if target in rolled:
                continue  # the new pod will read the current value anyway
            starts = workload_pod_starts(new_by_key[target], namespace)
            if starts and min(starts) < changed_at:
                stale.setdefault(target, []).append(("Secret", name))
                plan.stale_secret_detail.setdefault(target, []).append(
                    f"{name} written {changed_at:%Y-%m-%dT%H:%M:%SZ}, "
                    f"pod started {min(starts):%Y-%m-%dT%H:%M:%SZ}"
                )
    return stale


def doc_key(doc: dict) -> tuple:
    meta = doc.get("metadata") or {}
    return (doc.get("kind") or "", meta.get("name") or "")


def describe(key: tuple) -> str:
    return f"{key[0]}/{key[1]}"


def pod_template(doc: dict) -> dict | None:
    spec = doc.get("spec") or {}
    if doc.get("kind") == "CronJob":
        return ((spec.get("jobTemplate") or {}).get("spec") or {}).get("template")
    if doc.get("kind") in WORKLOAD_KINDS:
        return spec.get("template")
    return None


def config_refs(template: dict) -> tuple[set, set]:
    """Return the ConfigMap and Secret names a pod template consumes."""
    pod = template.get("spec") or {}
    configmaps: set = set()
    secrets: set = set()

    def add_volume_source(source: dict) -> None:
        if source.get("configMap"):
            configmaps.add(source["configMap"].get("name"))
        if source.get("secret"):
            secret = source["secret"]
            secrets.add(secret.get("secretName") or secret.get("name"))

    for volume in pod.get("volumes") or []:
        add_volume_source(volume)
        for source in (volume.get("projected") or {}).get("sources") or []:
            add_volume_source(source)

    containers = (pod.get("containers") or []) + (pod.get("initContainers") or [])
    for container in containers:
        for env_from in container.get("envFrom") or []:
            if env_from.get("configMapRef"):
                configmaps.add(env_from["configMapRef"].get("name"))
            if env_from.get("secretRef"):
                secrets.add(env_from["secretRef"].get("name"))
        for env in container.get("env") or []:
            value_from = env.get("valueFrom") or {}
            if value_from.get("configMapKeyRef"):
                configmaps.add(value_from["configMapKeyRef"].get("name"))
            if value_from.get("secretKeyRef"):
                secrets.add(value_from["secretKeyRef"].get("name"))

    configmaps.discard(None)
    secrets.discard(None)
    return configmaps, secrets


# --------------------------------------------------------------------------
# convergence plan
# --------------------------------------------------------------------------


def doc_lines(doc: dict) -> list[str]:
    """A doc as comparable lines, with config payloads expanded.

    A ConfigMap holding a 40 KB dashboard JSON is a single `data` string, so a
    naive serialization diffs it as one changed line and reports nothing about
    what moved inside it. Expanding each `data` entry into its own lines makes
    the counts mean what a reader expects. Secret *values* are never expanded --
    only their key names -- so a diff can never print a credential.
    """
    if doc.get("kind") == "Secret":
        skeleton = {k: v for k, v in doc.items() if k not in ("data", "stringData")}
        lines = json.dumps(skeleton, indent=2, sort_keys=True).splitlines()
        for field in ("data", "stringData"):
            for key, value in sorted((doc.get(field) or {}).items()):
                # A length, not the value: enough to see a rotation happened.
                lines.append(f"{field}/{key}: <{len(str(value))} bytes>")
        return lines

    payload = doc.get("data") if doc.get("kind") == "ConfigMap" else None
    if not isinstance(payload, dict):
        return json.dumps(doc, indent=2, sort_keys=True).splitlines()

    skeleton = {k: v for k, v in doc.items() if k != "data"}
    lines = json.dumps(skeleton, indent=2, sort_keys=True).splitlines()
    for key, value in sorted(payload.items()):
        lines.append(f"data/{key}:")
        lines.extend(f"  {line}" for line in str(value).splitlines())
    return lines


def doc_diff(live_doc: dict, new_doc: dict) -> tuple[int, int, list[str]]:
    """(added, removed, unified diff) between two versions of one resource."""
    before = doc_lines(live_doc)
    after = doc_lines(new_doc)
    hunks = list(difflib.unified_diff(before, after, lineterm="", n=1))
    added = sum(1 for line in hunks if line.startswith("+") and line[:3] != "+++")
    removed = sum(1 for line in hunks if line.startswith("-") and line[:3] != "---")
    return added, removed, hunks


def dropped_config_keys(live_doc: dict, new_doc: dict) -> list[tuple[str, int]]:
    """Config keys present live but absent from the render, with their size.

    This is the shape of an accidental revert: something was applied to the
    cluster that the chart does not produce, and this deploy would delete it. It
    reads as an ordinary "change" in a resource-level diff.
    """
    dropped = []
    for field in ("data", "binaryData", "stringData"):
        live_keys = live_doc.get(field) or {}
        new_keys = new_doc.get(field) or {}
        if not isinstance(live_keys, dict) or not isinstance(new_keys, dict):
            continue
        for key in sorted(set(live_keys) - set(new_keys)):
            size = len(str(live_keys[key]).splitlines())
            dropped.append((f"{field}/{key}", size))
    return dropped


def source_revision(doc: dict) -> str:
    """The git revision this workload's init container will check out, if any."""
    template = pod_template(doc)
    if template is None:
        return ""
    annotations = (template.get("metadata") or {}).get("annotations") or {}
    return str(annotations.get(SOURCE_REVISION_ANNOTATION) or "").strip()


def clone_mount_path(doc: dict) -> str | None:
    """Where the cloning init container's volume is mounted in the app container.

    Derived from the manifest rather than hardcoded, so moving the checkout in the
    chart does not silently disable the post-deploy revision check.
    """
    template = pod_template(doc)
    if template is None:
        return None
    spec = template.get("spec") or {}
    init_containers = spec.get("initContainers") or []
    cloners = [c for c in init_containers if c.get("name") == CLONE_INIT_CONTAINER]
    volume_names = {
        mount.get("name")
        for container in (cloners or init_containers)
        for mount in (container.get("volumeMounts") or [])
    }
    for container in spec.get("containers") or []:
        for mount in container.get("volumeMounts") or []:
            if mount.get("name") in volume_names:
                return mount.get("mountPath")
    return None


def try_output_line(args: list[str]) -> str | None:
    """`try_output` for a command whose output is a single token.

    It deliberately does not strip -- `helm get manifest` needs the raw bytes --
    so callers that interpolate the result into a message must, or the trailing
    newline lands mid-sentence.
    """
    raw = try_output(args)
    return None if raw is None else raw.strip()


def git_commit_exists(revision: str) -> bool:
    result = run(
        ["git", "-C", str(ROOT_DIR), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=False,
        quiet=True,
    )
    return result.returncode == 0


def source_revision_move(old: str, new: str) -> tuple[str, str]:
    """Classify a change of pinned revision as (verdict, detail).

    Verdicts: ``forward`` (new descends from old -- an ordinary upgrade),
    ``backward`` (old descends from new -- this ships older code than prod is
    running), ``diverged`` (neither, e.g. a rebased or abandoned branch), and
    ``unknown`` when the local clone cannot answer, which must not be treated as
    approval.
    """
    if shutil.which("git") is None:
        return "unknown", "git not available"
    if not (ROOT_DIR / ".git").exists():
        return "unknown", f"{ROOT_DIR} is not a git clone"
    missing = [rev for rev in (old, new) if not git_commit_exists(rev)]
    if missing:
        return "unknown", (
            "not in the local clone: " + ", ".join(missing) + " (try: git fetch origin)"
        )
    if (
        run(
            ["git", "-C", str(ROOT_DIR), "merge-base", "--is-ancestor", old, new],
            check=False,
            quiet=True,
        ).returncode
        == 0
    ):
        count = try_output_line(
            ["git", "-C", str(ROOT_DIR), "rev-list", "--count", f"{old}..{new}"]
        )
        return "forward", f"{count or '?'} commits forward"
    if (
        run(
            ["git", "-C", str(ROOT_DIR), "merge-base", "--is-ancestor", new, old],
            check=False,
            quiet=True,
        ).returncode
        == 0
    ):
        count = try_output_line(
            ["git", "-C", str(ROOT_DIR), "rev-list", "--count", f"{new}..{old}"]
        )
        return "backward", f"drops {count or '?'} commits that are live"
    return "diverged", "neither revision contains the other"


class Plan:
    def __init__(self) -> None:
        self.created: list = []
        self.changed: list = []
        self.removed: list = []
        self.rolled_by_helm: list = []
        self.restart: dict = {}
        self.reload: dict = {}
        self.self_applying: dict = {}
        self.cronjobs: dict = {}
        self.stale_secrets: dict = {}
        self.stale_secret_detail: dict = {}
        self.unconsumed: list = []
        self.workloads: list = []
        self.replicas: dict = {}
        self.first_install = False
        # key -> (added, removed, unified diff lines)
        self.diffs: dict = {}
        # key -> [(config key, line count)] present live, absent from the render
        self.dropped: dict = {}
        # key -> (old revision, new revision, verdict, detail)
        self.source_moves: dict = {}
        # the rendered docs, so verification can re-read what was asked for
        self.new_by_key: dict = {}


def build_plan(
    live_docs: list[dict], new_docs: list[dict], first_install: bool
) -> Plan:
    plan = Plan()
    plan.first_install = first_install

    live = {doc_key(d): d for d in live_docs}
    new = {doc_key(d): d for d in new_docs}
    plan.new_by_key = new

    plan.created = sorted(k for k in new if k not in live)
    plan.removed = sorted(k for k in live if k not in new)
    plan.changed = sorted(k for k in new if k in live and new[k] != live[k])

    for key in plan.changed:
        plan.diffs[key] = doc_diff(live[key], new[key])
        dropped = dropped_config_keys(live[key], new[key])
        if dropped:
            plan.dropped[key] = dropped

    # Pinned-revision moves, for both changed and unchanged workloads: an
    # unchanged pin is silent, but a changed one has to be shown to be judged.
    for key in plan.changed:
        if key[0] not in WORKLOAD_KINDS:
            continue
        old_rev = source_revision(live[key])
        new_rev = source_revision(new[key])
        if not (old_rev or new_rev) or old_rev == new_rev:
            continue
        if old_rev and new_rev:
            verdict, detail = source_revision_move(old_rev, new_rev)
        else:
            # One side unpinned: the clone follows the default branch there, so
            # there is no commit pair to compare.
            verdict, detail = "unpinned", "one side tracks the default branch"
        plan.source_moves[key] = (old_rev, new_rev, verdict, detail)

    for key in sorted(new):
        if key[0] in WORKLOAD_KINDS:
            plan.workloads.append(key)
            plan.replicas[key] = (new[key].get("spec") or {}).get("replicas")

    if first_install:
        # Everything is being created; nothing is running on stale config.
        return plan

    # Workloads Helm will roll on its own: their pod template changed.
    rolled = set()
    for key in plan.changed + plan.created:
        if key[0] not in WORKLOAD_KINDS:
            continue
        old_template = pod_template(live[key]) if key in live else None
        if pod_template(new[key]) != old_template:
            rolled.add(key)
            plan.rolled_by_helm.append(key)

    # Map each changed config object to the workloads that consume it.
    changed_configs = [k for k in plan.changed if k[0] in CONFIG_KINDS]
    consumers: dict = {}
    for key, doc in new.items():
        template = pod_template(doc)
        if template is None:
            continue
        configmaps, secrets = config_refs(template)
        for name in configmaps:
            consumers.setdefault(("ConfigMap", name), []).append(key)
        for name in secrets:
            consumers.setdefault(("Secret", name), []).append(key)

    for config in changed_configs:
        targets = consumers.get(config)
        if not targets:
            plan.unconsumed.append(config)
            continue
        for target in sorted(targets):
            if target in rolled:
                continue  # Helm rolls it; the new config comes up with the pod.
            if target[0] == "CronJob":
                plan.cronjobs.setdefault(target, []).append(config)
                continue
            how = CONFIG_APPLY.get(config[1], APPLY_RESTART)
            if how == APPLY_AUTO:
                plan.self_applying.setdefault(target, []).append(config)
            elif how == APPLY_RELOAD and target in RELOAD_COMMANDS:
                plan.reload.setdefault(target, []).append(config)
            else:
                plan.restart.setdefault(target, []).append(config)

    # A restart supersedes a reload for the same workload.
    for target in list(plan.reload):
        if target in plan.restart:
            plan.restart[target].extend(plan.reload.pop(target))

    return plan


SOURCE_VERDICT_NOTE = {
    "forward": "",
    "backward": "  !! BACKWARDS -- ships older code than prod is running",
    "diverged": "  !! DIVERGED -- neither revision contains the other",
    "unknown": "  !! UNVERIFIED",
    "unpinned": "",
}


def source_move_blockers(plan: Plan) -> list[str]:
    """Pinned-revision moves that should stop a deploy until acknowledged."""
    blockers = []
    for key, (old_rev, new_rev, verdict, detail) in sorted(plan.source_moves.items()):
        if verdict in ("forward", "unpinned"):
            continue
        blockers.append(
            f"{describe(key)}: {short_rev(old_rev)} -> {short_rev(new_rev)} "
            f"({verdict}: {detail})"
        )
    return blockers


def short_rev(revision: str) -> str:
    return revision[:9] if len(revision) > 12 else revision or "(unpinned)"


def print_plan(plan: Plan, revision: str | None, show_diff: bool = False) -> None:
    where = f"live revision {revision}" if revision else "no existing release"
    note(f"\n=== change plan ({where}) ===")
    if plan.first_install:
        note(f"  first install: creating {len(plan.created)} resources")
    elif not (plan.created or plan.changed or plan.removed):
        note("  no manifest changes (upgrade only bumps the release revision)")
    else:
        for label, keys in (
            ("create", plan.created),
            ("change", plan.changed),
            ("delete", plan.removed),
        ):
            for key in keys:
                counts = ""
                if key in plan.diffs:
                    added, removed, _ = plan.diffs[key]
                    counts = f"  +{added} -{removed} lines"
                note(f"  {label:<7} {describe(key)}{counts}")

    if plan.dropped:
        # Loud, because a resource-level diff renders this as an ordinary
        # "change" and it is how hand-applied config gets silently reverted.
        note(
            "\n  !! this deploy DELETES config that is live but absent from the render:"
        )
        for key in sorted(plan.dropped):
            for config_key, size in plan.dropped[key]:
                note(f"       {describe(key)}  {config_key}  ({size} lines)")
        note("     nothing in the chart produces these -- check they are not")
        note("     someone's live change before applying.")

    if plan.source_moves:
        note("\n=== cloned source revisions (checked out at pod start) ===")
        for key in sorted(plan.source_moves):
            old_rev, new_rev, verdict, detail = plan.source_moves[key]
            note(
                f"  {describe(key)}  {short_rev(old_rev)} -> {short_rev(new_rev)}"
                f"  ({detail}){SOURCE_VERDICT_NOTE.get(verdict, '')}"
            )

    if show_diff:
        for key in sorted(plan.diffs):
            _added, _removed, hunks = plan.diffs[key]
            if not hunks:
                continue
            note(f"\n--- diff {describe(key)} ---")
            for line in hunks:
                note(f"  {line}")

    note("\n=== pod convergence plan ===")

    def line(label: str, targets: dict) -> None:
        if not targets:
            return
        note(f"  {label}")
        for target in dependency_order(targets):
            configs = targets[target]
            reasons = ", ".join(sorted({c[1] for c in configs}))
            replicas = plan.replicas.get(target)
            warn = ""
            if (
                target in plan.restart or target in plan.stale_secrets
            ) and replicas == 1:
                warn = "  [single replica: brief gap]"
            note(f"    {describe(target)}  <- {reasons}{warn}")

    if plan.rolled_by_helm:
        note("  helm rolls these (pod template changed)")
        for key in sorted(plan.rolled_by_helm):
            note(f"    {describe(key)}")
    line("restart needed (config read only at startup)", plan.restart)
    if plan.stale_secrets:
        note("  restart needed (Secret written after the pod started)")
        for target in dependency_order(plan.stale_secrets):
            replicas = plan.replicas.get(target)
            warn = "  [single replica: brief gap]" if replicas == 1 else ""
            note(f"    {describe(target)}{warn}")
            for detail in plan.stale_secret_detail.get(target, []):
                note(f"      {detail}")
    line("reload in place (no restart)", plan.reload)
    line("no action, process reloads itself", plan.self_applying)
    line("next scheduled run picks it up", plan.cronjobs)
    if plan.unconsumed:
        note("  changed but nothing mounts it")
        for key in plan.unconsumed:
            note(f"    {describe(key)}")
    if not (
        plan.rolled_by_helm
        or plan.restart
        or plan.reload
        or plan.self_applying
        or plan.cronjobs
        or plan.stale_secrets
    ):
        note("  nothing to restart")
    note("")


# --------------------------------------------------------------------------
# convergence + verification
# --------------------------------------------------------------------------


def pod_snapshot(namespace: str, release: str) -> dict:
    """Map pod name -> (ready, total restarts) for the release's pods."""
    raw = try_output(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app.kubernetes.io/instance={release}",
            "-o",
            "json",
        ]
    )
    if raw is None:
        return {}
    snapshot = {}
    for pod in json.loads(raw).get("items") or []:
        statuses = (pod.get("status") or {}).get("containerStatuses") or []
        restarts = sum(s.get("restartCount") or 0 for s in statuses)
        snapshot[pod["metadata"]["name"]] = restarts
    return snapshot


def dependency_order(targets) -> list:
    """Sort workloads so a dependency is restarted before its consumers."""

    def sort_key(target: tuple) -> tuple:
        name = target[1]
        if name in RESTART_ORDER:
            return (RESTART_ORDER.index(name), name)
        return (len(RESTART_ORDER), name)

    return sorted(targets, key=sort_key)


def converge(plan: Plan, args: argparse.Namespace) -> bool:
    """Restart or reload the workloads a config-only change left behind.

    Runs strictly one workload at a time, in dependency order, waiting for each
    to become healthy before touching the next. If a dependency fails to come
    back the rest is abandoned rather than restarted on top of a broken backend.
    """
    skip = {s.strip() for s in (args.skip_restart or "").split(",") if s.strip()}

    if args.restart_all:
        restarts = {key: [] for key in plan.workloads}
    else:
        restarts = dict(plan.restart)
        for target, secrets in plan.stale_secrets.items():
            restarts.setdefault(target, []).extend(secrets)

    actions = {t: APPLY_RESTART for t in restarts}
    for target in plan.reload:
        actions.setdefault(target, APPLY_RELOAD)
    if not actions:
        return True

    ordered = dependency_order(actions)
    note("\n=== converge (dependency order) ===")
    note("  " + " -> ".join(describe(t) for t in ordered))

    for target in ordered:
        how = actions[target]
        configs = (
            restarts.get(target) if how == APPLY_RESTART else plan.reload.get(target)
        )
        reasons = ", ".join(sorted({c[1] for c in configs or []})) or "--restart-all"
        if target[1] in skip:
            note(f"skipping {how} of {describe(target)} (--skip-restart)")
            continue

        if how == APPLY_RELOAD:
            note(f"reloading {describe(target)} in place ({reasons})")
            if reload_workload(target, args):
                continue
            note(f"  reload failed; falling back to a restart of {describe(target)}")

        note(f"restarting {describe(target)} ({reasons})")
        resource = f"{target[0].lower()}/{target[1]}"
        if (
            run(
                ["kubectl", "rollout", "restart", resource, "-n", args.namespace],
                check=False,
            ).returncode
            != 0
        ):
            note(
                f"  FAILED to restart {resource} -- stopping, later workloads depend on it"
            )
            return False
        if (
            run(
                [
                    "kubectl",
                    "rollout",
                    "status",
                    resource,
                    "-n",
                    args.namespace,
                    f"--timeout={args.timeout}",
                ],
                check=False,
            ).returncode
            != 0
        ):
            note(f"  {resource} did NOT come back -- stopping before its consumers")
            return False

    return True


def reload_workload(target: tuple, args: argparse.Namespace) -> bool:
    container, command = RELOAD_COMMANDS[target]
    pods = try_output(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            args.namespace,
            "-l",
            f"app.kubernetes.io/instance={args.release}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}",
        ]
    )
    if not pods:
        return False
    prefix = f"{target[1]}-"
    matches = [p for p in pods.split() if p == target[1] or p.startswith(prefix)]
    if not matches:
        return False
    for pod in matches:
        result = run(
            ["kubectl", "exec", "-n", args.namespace, pod, "-c", container, "--"]
            + command,
            check=False,
            quiet=True,
        )
        if result.returncode != 0:
            return False
    return True


def verify_source_revisions(
    plan: Plan, new_by_key: dict, args: argparse.Namespace
) -> bool:
    """Confirm each cloning workload is actually running its pinned revision.

    A rollout completing proves the pod template applied, not that the code half
    of the deploy did anything: the source lives in an emptyDir populated by an
    init container, so a pin that did not change leaves the old commit in place
    while `helm upgrade` reports success. Read the checked-out revision back out
    of the pod instead of trusting the annotation.
    """
    ok = True
    for key in dependency_order({k: None for k in plan.workloads}):
        doc = new_by_key.get(key)
        if doc is None or key[0] not in ("Deployment", "StatefulSet"):
            continue
        pinned = source_revision(doc)
        mount = clone_mount_path(doc)
        if not pinned or not mount:
            continue
        container = ((pod_template(doc) or {}).get("spec") or {}).get("containers")
        container_name = (container or [{}])[0].get("name") or key[1]
        head = try_output_line(
            [
                "kubectl",
                "exec",
                f"{key[0].lower()}/{key[1]}",
                "-n",
                args.namespace,
                "-c",
                container_name,
                "--",
                "python",
                "-c",
                f"print(open({mount!r} + '/.git/HEAD').read().strip())",
            ]
        )
        if head is None:
            # Not a failure: the image may have no python, or the path may have
            # moved. Say so rather than implying the revision was confirmed.
            note(f"  {describe(key)}: could not read {mount}/.git/HEAD (unverified)")
            continue
        if head.startswith("ref: "):
            note(f"  {describe(key)}: on branch {head[5:]} (pin: {short_rev(pinned)})")
            continue
        if head.startswith(pinned) or pinned.startswith(head):
            note(f"  {describe(key)}: running {short_rev(head)} as pinned")
        else:
            note(
                f"  {describe(key)}: running {short_rev(head)} but pinned to "
                f"{short_rev(pinned)} -- the code half of this deploy did NOT land"
            )
            ok = False
    return ok


def verify(plan: Plan, args: argparse.Namespace, baseline: dict) -> bool:
    """Wait for every workload to roll out, then check the pods are healthy."""
    note("\n=== verify ===")
    ok = True

    for kind, name in dependency_order(plan.workloads):
        resource = f"{kind.lower()}/{name}"
        result = run(
            [
                "kubectl",
                "rollout",
                "status",
                resource,
                "-n",
                args.namespace,
                f"--timeout={args.timeout}",
            ],
            check=False,
        )
        if result.returncode != 0:
            note(f"  {resource}: rollout did NOT complete")
            ok = False

    raw = try_output(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            args.namespace,
            "-l",
            f"app.kubernetes.io/instance={args.release}",
            "-o",
            "json",
        ]
    )
    if raw is None:
        note("  could not list pods")
        return False

    for pod in sorted(
        json.loads(raw).get("items") or [], key=lambda p: p["metadata"]["name"]
    ):
        name = pod["metadata"]["name"]
        status = pod.get("status") or {}
        phase = status.get("phase")
        if phase in ("Succeeded", "Failed"):
            continue  # completed CronJob pod, not part of the running stack
        statuses = status.get("containerStatuses") or []
        not_ready = [s["name"] for s in statuses if not s.get("ready")]
        restarts = sum(s.get("restartCount") or 0 for s in statuses)
        new_restarts = restarts - baseline.get(name, 0)
        problems = []
        if phase != "Running":
            problems.append(f"phase={phase}")
        if not_ready:
            problems.append("not ready: " + ",".join(not_ready))
        for state in statuses:
            reason = ((state.get("state") or {}).get("waiting") or {}).get("reason")
            if reason in BAD_WAITING_REASONS:
                problems.append(f"{state['name']}: {reason}")
        # A pod created by this deploy has no baseline, so only count restarts
        # for pods that already existed -- those indicate a crash we caused.
        if name in baseline and new_restarts > 0:
            problems.append(f"{new_restarts} new container restart(s)")
        if problems:
            note(f"  {name}: " + "; ".join(problems))
            ok = False
        else:
            note(f"  {name}: ready ({restarts} lifetime restarts)")

    if not args.no_source_check:
        ok = verify_source_revisions(plan, plan.new_by_key, args) and ok

    note("  OK: all workloads rolled out and pods are ready" if ok else "  FAILED")
    return ok


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy/scripts/deploy.py",
        description=(
            "Deploy the transformers-ci Helm chart. The committed env/example.yaml "
            "carries public-safe placeholders. For a real deployment pass a private "
            "values file with -f."
        ),
    )
    parser.add_argument(
        "-n",
        "--namespace",
        default="transformers-ci",
        help="Kubernetes namespace (default: transformers-ci)",
    )
    parser.add_argument(
        "-r",
        "--release",
        default="transformers-ci",
        help="Helm release name (default: transformers-ci)",
    )
    parser.add_argument(
        "-f",
        "--values",
        default=str(CHART_DIR / "env" / "example.yaml"),
        help="Helm values file (default: deploy/helm/env/example.yaml)",
    )
    parser.add_argument(
        "--secret-file", help="Apply a local Secret manifest before deploying"
    )
    parser.add_argument(
        "--secret-name",
        default="transformers-ci-secrets",
        help="Secret name to clean up after apply (default: transformers-ci-secrets)",
    )
    parser.add_argument(
        "--context", help="Require this kubectl context before deploying"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render manifests to stdout (plan goes to stderr) without changing the cluster",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show the change and pod-restart plan only, without deploying",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Deploy and verify, but do not restart/reload workloads left on stale config",
    )
    parser.add_argument(
        "--restart-all",
        action="store_true",
        help="Restart every workload regardless of what changed",
    )
    parser.add_argument(
        "--skip-restart",
        help="Comma-separated workload names to leave alone (e.g. tempo)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-deploy rollout and pod health checks",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print the actual manifest diff for each changed resource",
    )
    parser.add_argument(
        "--allow-source-rollback",
        action="store_true",
        help=(
            "Apply even when a pinned sourceRevision moves backwards or to a "
            "diverged commit (default: refuse, since it ships older code than "
            "prod is running)"
        ),
    )
    parser.add_argument(
        "--no-source-check",
        action="store_true",
        help=(
            "Skip reading each cloning workload's checked-out revision back out "
            "of its pod after the deploy"
        ),
    )
    parser.add_argument(
        "--no-secret-check",
        action="store_true",
        help=(
            "Skip detection of pods running a rotated Secret (compares Secret "
            "write time against pod start time; reads Secret metadata only)"
        ),
    )
    parser.add_argument(
        "--timeout",
        default="10m",
        help="Timeout for helm --wait and each rollout status (default: 10m)",
    )
    return parser


def render_chart(args: argparse.Namespace, values_file: Path) -> str:
    return output(
        [
            "helm",
            "template",
            args.release,
            str(CHART_DIR),
            "-n",
            args.namespace,
            "-f",
            str(values_file),
        ]
    )


def load_plan(
    args: argparse.Namespace, new_docs: list[dict]
) -> tuple[Plan, str | None]:
    """Diff the render against the live release. Never fatal: the deploy can
    still proceed without a plan, it just cannot skip needless restarts."""
    live_manifest = try_output(
        ["helm", "get", "manifest", args.release, "-n", args.namespace]
    )
    revision = None
    if live_manifest is not None:
        revision = try_output(
            [
                "helm",
                "status",
                args.release,
                "-n",
                args.namespace,
                "-o",
                "json",
            ]
        )
        if revision:
            try:
                revision = str(json.loads(revision).get("version"))
            except (ValueError, AttributeError):
                revision = None
    plan = build_plan(
        manifest_to_docs(live_manifest or ""),
        new_docs,
        first_install=live_manifest is None,
    )
    if not plan.first_install and not args.no_secret_check:
        plan.stale_secrets = find_stale_secret_consumers(
            plan, {doc_key(d): d for d in new_docs}, args.namespace
        )
    return plan, revision


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    require_cmd("kubectl")
    require_cmd("helm")

    values_file = Path(args.values)
    if not CHART_DIR.is_dir():
        raise SystemExit(f"chart directory not found: {CHART_DIR}")
    if not values_file.is_file():
        raise SystemExit(f"values file not found: {values_file}")

    current_context = output(["kubectl", "config", "current-context"])
    if args.context and current_context != args.context:
        raise SystemExit(
            f"refusing to deploy to context '{current_context}' (expected '{args.context}')"
        )

    print(f"Context: {current_context}")
    print(f"Namespace: {args.namespace}")
    print(f"Release: {args.release}")
    print(f"Values: {values_file}")
    # stdout is block-buffered when piped; flush so this header cannot land after
    # the plan and progress output that follow it on stderr.
    sys.stdout.flush()

    # Chart integrity first: a checkout whose dashboard symlinks did not survive
    # renders fine and deploys garbage, so refuse before touching the cluster.
    check_chart_files()

    rendered = render_chart(args, values_file)
    new_docs = manifest_to_docs(rendered)
    validate_rendered_dashboards(new_docs)

    if args.dry_run or args.plan:
        if args.dry_run:
            print(rendered)
        plan, revision = load_plan(args, new_docs)
        print_plan(plan, revision, show_diff=args.diff)
        return 0

    plan, revision = load_plan(args, new_docs)
    print_plan(plan, revision, show_diff=args.diff)

    # A pinned revision moving anywhere other than forward is refused rather than
    # warned about: it means the values file disagrees with what prod is running,
    # and applying it rolls the code back while every other signal says success.
    blockers = source_move_blockers(plan)
    if blockers and not args.allow_source_rollback:
        for blocker in blockers:
            note(f"\nrefusing to apply: {blocker}")
        raise SystemExit(
            "pinned source revision does not move forward -- reconcile the values "
            "file with the live release, or pass --allow-source-rollback"
        )

    namespace_exists = (
        run(
            ["kubectl", "get", "namespace", args.namespace], check=False, quiet=True
        ).returncode
        == 0
    )
    if not namespace_exists:
        run(["kubectl", "create", "namespace", args.namespace])

    if args.secret_file:
        secret_file = Path(args.secret_file)
        if not secret_file.is_file():
            raise SystemExit(f"secret file not found: {secret_file}")
        run(["kubectl", "apply", "-n", args.namespace, "-f", str(secret_file)])
        run(
            [
                "kubectl",
                "annotate",
                "secret",
                args.secret_name,
                "-n",
                args.namespace,
                "kubectl.kubernetes.io/last-applied-configuration-",
            ],
            check=False,
            quiet=True,
        )

    baseline = pod_snapshot(args.namespace, args.release)

    run(
        [
            "helm",
            "upgrade",
            "--install",
            args.release,
            str(CHART_DIR),
            "-n",
            args.namespace,
            "-f",
            str(values_file),
            "--wait",
            "--timeout",
            args.timeout,
        ]
    )

    converged = True
    if args.no_restart:
        pending = sorted(set(plan.restart) | set(plan.reload))
        if pending:
            note("\n--no-restart: these workloads are still on their old config:")
            for target in pending:
                note(f"  {describe(target)}")
    else:
        converged = converge(plan, args)

    verified = True
    if not args.no_verify:
        verified = verify(plan, args, baseline)

    if not (converged and verified):
        note("\ndeploy finished with problems -- see above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

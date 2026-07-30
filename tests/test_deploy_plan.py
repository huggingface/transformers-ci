"""Tests for the restart planner in deploy/scripts/deploy.py.

The planner decides which pods a deploy has to restart. Both ways of being wrong
are silent in production: a missed restart ships a deploy that changes nothing
(the ConfigMap updates, the process keeps its old config), and a spurious restart
costs an avoidable outage window on the single-replica workloads.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "scripts" / "deploy.py"


def load_deploy():
    spec = importlib.util.spec_from_file_location("deploy_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy = load_deploy()


def configmap(name: str, body: str = "a") -> dict:
    return {"kind": "ConfigMap", "metadata": {"name": name}, "data": {"config": body}}


def workload(
    kind: str,
    name: str,
    *,
    configmaps=(),
    secrets=(),
    image: str = "img:1",
    replicas: int = 1,
) -> dict:
    volumes = [{"name": c, "configMap": {"name": c}} for c in configmaps]
    volumes += [{"name": s, "secret": {"secretName": s}} for s in secrets]
    template = {
        "metadata": {"annotations": {}},
        "spec": {"containers": [{"name": name, "image": image}], "volumes": volumes},
    }
    spec = {"replicas": replicas, "template": template}
    if kind == "CronJob":
        spec = {"jobTemplate": {"spec": {"template": template}}}
    return {"kind": kind, "metadata": {"name": name}, "spec": spec}


def plan_for(live: list, new: list):
    return deploy.build_plan(live, new, first_install=False)


def names(targets) -> set:
    return {name for _, name in targets}


class TestConfigChangeRestarts:
    def test_configmap_change_restarts_its_consumer(self):
        """The whole point: a ConfigMap-only change leaves the process stale."""
        live = [
            configmap("tempo-config", "old"),
            workload("StatefulSet", "tempo", configmaps=["tempo-config"]),
        ]
        new = [
            configmap("tempo-config", "new"),
            workload("StatefulSet", "tempo", configmaps=["tempo-config"]),
        ]
        plan = plan_for(live, new)
        assert names(plan.restart) == {"tempo"}
        assert plan.rolled_by_helm == []

    def test_unchanged_configmap_restarts_nothing(self):
        live = [
            configmap("tempo-config"),
            workload("StatefulSet", "tempo", configmaps=["tempo-config"]),
        ]
        plan = plan_for(live, list(live))
        assert plan.restart == {}
        assert plan.changed == []

    def test_pod_template_change_is_left_to_helm(self):
        """Helm already rolls the pod, so restarting it again is wasted downtime."""
        live = [
            configmap("tempo-config", "old"),
            workload(
                "StatefulSet", "tempo", configmaps=["tempo-config"], image="img:1"
            ),
        ]
        new = [
            configmap("tempo-config", "new"),
            workload(
                "StatefulSet", "tempo", configmaps=["tempo-config"], image="img:2"
            ),
        ]
        plan = plan_for(live, new)
        assert plan.restart == {}
        assert names(plan.rolled_by_helm) == {"tempo"}

    def test_replica_count_change_alone_does_not_restart(self):
        live = [workload("Deployment", "otelcol", replicas=2)]
        new = [workload("Deployment", "otelcol", replicas=3)]
        plan = plan_for(live, new)
        assert plan.restart == {}
        assert plan.rolled_by_helm == []

    def test_secret_change_restarts_its_consumer(self):
        live = [
            {"kind": "Secret", "metadata": {"name": "creds"}, "data": {"k": "old"}},
            workload("Deployment", "trace-exporter", secrets=["creds"]),
        ]
        new = [
            {"kind": "Secret", "metadata": {"name": "creds"}, "data": {"k": "new"}},
            workload("Deployment", "trace-exporter", secrets=["creds"]),
        ]
        assert names(plan_for(live, new).restart) == {"trace-exporter"}

    def test_one_configmap_restarts_every_consumer(self):
        shared = ["shared-config"]
        live = [configmap("shared-config", "old")] + [
            workload("Deployment", n, configmaps=shared) for n in ("grafana", "otelcol")
        ]
        new = [configmap("shared-config", "new")] + [
            workload("Deployment", n, configmaps=shared) for n in ("grafana", "otelcol")
        ]
        assert names(plan_for(live, new).restart) == {"grafana", "otelcol"}


class TestApplyPolicies:
    def test_self_reloading_config_needs_no_restart(self):
        """Grafana's file provisioner rescans dashboards on its own."""
        live = [
            configmap("grafana-dashboards", "old"),
            workload("Deployment", "grafana", configmaps=["grafana-dashboards"]),
        ]
        new = [
            configmap("grafana-dashboards", "new"),
            workload("Deployment", "grafana", configmaps=["grafana-dashboards"]),
        ]
        plan = plan_for(live, new)
        assert plan.restart == {}
        assert plan.reload == {}
        assert names(plan.self_applying) == {"grafana"}

    def test_prometheus_config_reloads_instead_of_restarting(self):
        live = [
            configmap("prometheus-config", "old"),
            workload("StatefulSet", "prometheus", configmaps=["prometheus-config"]),
        ]
        new = [
            configmap("prometheus-config", "new"),
            workload("StatefulSet", "prometheus", configmaps=["prometheus-config"]),
        ]
        plan = plan_for(live, new)
        assert names(plan.reload) == {"prometheus"}
        assert plan.restart == {}

    def test_restart_supersedes_reload_for_same_workload(self):
        """If anything about the workload needs a restart, reloading is redundant."""
        live = [
            configmap("prometheus-config", "old"),
            configmap("other-config", "old"),
            workload(
                "StatefulSet",
                "prometheus",
                configmaps=["prometheus-config", "other-config"],
            ),
        ]
        new = [
            configmap("prometheus-config", "new"),
            configmap("other-config", "new"),
            workload(
                "StatefulSet",
                "prometheus",
                configmaps=["prometheus-config", "other-config"],
            ),
        ]
        plan = plan_for(live, new)
        assert names(plan.restart) == {"prometheus"}
        assert plan.reload == {}

    def test_unknown_config_defaults_to_restart(self):
        live = [
            configmap("brand-new-thing", "old"),
            workload("Deployment", "widget", configmaps=["brand-new-thing"]),
        ]
        new = [
            configmap("brand-new-thing", "new"),
            workload("Deployment", "widget", configmaps=["brand-new-thing"]),
        ]
        assert names(plan_for(live, new).restart) == {"widget"}


class TestNonRestartableTargets:
    def test_cronjob_is_never_restarted(self):
        live = [
            configmap("job-config", "old"),
            workload("CronJob", "ci-data-publisher", configmaps=["job-config"]),
        ]
        new = [
            configmap("job-config", "new"),
            workload("CronJob", "ci-data-publisher", configmaps=["job-config"]),
        ]
        plan = plan_for(live, new)
        assert plan.restart == {}
        assert names(plan.cronjobs) == {"ci-data-publisher"}

    def test_configmap_nobody_mounts_is_reported(self):
        live = [configmap("orphan", "old")]
        new = [configmap("orphan", "new")]
        plan = plan_for(live, new)
        assert plan.restart == {}
        assert plan.unconsumed == [("ConfigMap", "orphan")]

    def test_first_install_restarts_nothing(self):
        new = [
            configmap("tempo-config"),
            workload("StatefulSet", "tempo", configmaps=["tempo-config"]),
        ]
        plan = deploy.build_plan([], new, first_install=True)
        assert plan.restart == {}
        assert plan.reload == {}
        assert names(plan.workloads) == {"tempo"}


class TestOrdering:
    def test_dependencies_restart_before_consumers(self):
        targets = [
            ("Deployment", "grafana"),
            ("Deployment", "otelcol"),
            ("StatefulSet", "tempo"),
            ("StatefulSet", "prometheus"),
        ]
        assert [n for _, n in deploy.dependency_order(targets)] == [
            "tempo",
            "otelcol",
            "prometheus",
            "grafana",
        ]

    def test_tempo_precedes_otelcol(self):
        """A tempo restart must not overlap otelcol's, or exports are lost."""
        order = [
            n
            for _, n in deploy.dependency_order(
                [("Deployment", "otelcol"), ("StatefulSet", "tempo")]
            )
        ]
        assert order.index("tempo") < order.index("otelcol")

    def test_unknown_workloads_sort_last_and_stably(self):
        order = [
            n
            for _, n in deploy.dependency_order(
                [
                    ("Deployment", "zebra"),
                    ("Deployment", "apple"),
                    ("StatefulSet", "tempo"),
                ]
            )
        ]
        assert order == ["tempo", "apple", "zebra"]


class TestSymlinkDetection:
    def test_detects_symlink_materialised_as_text(self, tmp_path):
        path = tmp_path / "dash.json"
        path.write_text("../../../dashboard/dash.json")
        assert deploy.unresolved_symlink_target(path) == "../../../dashboard/dash.json"

    def test_real_json_is_not_flagged(self, tmp_path):
        path = tmp_path / "dash.json"
        path.write_text('{"title": "a/b", "panels": []}')
        assert deploy.unresolved_symlink_target(path) is None

    def test_actual_symlink_is_not_flagged(self, tmp_path):
        target = tmp_path / "real.json"
        target.write_text("{}")
        link = tmp_path / "link.json"
        link.symlink_to(target)
        assert deploy.unresolved_symlink_target(link) is None

    def test_multiline_text_is_not_flagged(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("some/path\nsecond line\n")
        assert deploy.unresolved_symlink_target(path) is None

    def test_windows_style_checkout_is_rejected(self, tmp_path, monkeypatch):
        dashboards = tmp_path / "dashboards"
        dashboards.mkdir()
        (dashboards / "a.json").write_text("../../../dashboard/a.json")
        monkeypatch.setattr(deploy, "DASHBOARD_DIR", dashboards)
        with pytest.raises(SystemExit) as excinfo:
            deploy.check_chart_files()
        assert "core.symlinks" in str(excinfo.value)

    def test_healthy_checkout_passes(self, tmp_path, monkeypatch):
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.json").write_text('{"panels": []}')
        dashboards = tmp_path / "dashboards"
        dashboards.mkdir()
        (dashboards / "a.json").symlink_to(source / "a.json")
        monkeypatch.setattr(deploy, "DASHBOARD_DIR", dashboards)
        deploy.check_chart_files()

    def test_dangling_symlink_is_rejected(self, tmp_path, monkeypatch):
        dashboards = tmp_path / "dashboards"
        dashboards.mkdir()
        (dashboards / "a.json").symlink_to(tmp_path / "missing.json")
        monkeypatch.setattr(deploy, "DASHBOARD_DIR", dashboards)
        with pytest.raises(SystemExit) as excinfo:
            deploy.check_chart_files()
        assert "dangling" in str(excinfo.value)


class TestDashboardValidation:
    def test_invalid_dashboard_json_is_rejected(self):
        docs = [
            {
                "kind": "ConfigMap",
                "metadata": {"name": "grafana-dashboards"},
                "data": {"a.json": "../../../dashboard/a.json"},
            }
        ]
        with pytest.raises(SystemExit) as excinfo:
            deploy.validate_rendered_dashboards(docs)
        assert "invalid" in str(excinfo.value)

    def test_valid_dashboard_json_passes(self):
        docs = [
            {
                "kind": "ConfigMap",
                "metadata": {"name": "grafana-dashboards-restricted"},
                "data": {"a.json": '{"panels": []}', "VERSION": "1.2.3"},
            }
        ]
        deploy.validate_rendered_dashboards(docs)

    def test_other_configmaps_are_not_json_checked(self):
        docs = [
            {
                "kind": "ConfigMap",
                "metadata": {"name": "tempo-config"},
                "data": {"tempo.json": "not json at all"},
            }
        ]
        deploy.validate_rendered_dashboards(docs)


class TestRotatedSecretDetection:
    """A rotated secret value never shows up in a manifest diff, so staleness is
    inferred from Secret write time vs pod start time."""

    def setup_method(self):
        self.plan = deploy.Plan()

    def check(self, monkeypatch, *, written, pod_started, kind="Deployment"):
        doc = workload(kind, "trace-exporter", secrets=["creds"])
        doc["spec"]["selector"] = {"matchLabels": {"app": "trace-exporter"}}
        monkeypatch.setattr(
            deploy, "secret_write_times", lambda ns, names: {"creds": written}
        )
        monkeypatch.setattr(deploy, "workload_pod_starts", lambda d, ns: [pod_started])
        return deploy.find_stale_secret_consumers(
            self.plan, {(kind, "trace-exporter"): doc}, "ns"
        )

    def test_pod_older_than_secret_is_stale(self, monkeypatch):
        stale = self.check(
            monkeypatch,
            written=datetime(2026, 7, 30, 13, 0),
            pod_started=datetime(2026, 7, 29, 6, 0),
        )
        assert names(stale) == {"trace-exporter"}

    def test_pod_newer_than_secret_is_current(self, monkeypatch):
        stale = self.check(
            monkeypatch,
            written=datetime(2026, 7, 30, 13, 0),
            pod_started=datetime(2026, 7, 30, 15, 0),
        )
        assert stale == {}

    def test_workload_helm_will_roll_is_skipped(self, monkeypatch):
        """The new pod reads the current value, so restarting again is wasteful."""
        self.plan.rolled_by_helm = [("Deployment", "trace-exporter")]
        stale = self.check(
            monkeypatch,
            written=datetime(2026, 7, 30, 13, 0),
            pod_started=datetime(2026, 7, 29, 6, 0),
        )
        assert stale == {}

    def test_cronjobs_are_ignored(self, monkeypatch):
        """A CronJob reads the Secret fresh on its next scheduled run."""
        doc = workload("CronJob", "publisher", secrets=["creds"])
        monkeypatch.setattr(
            deploy,
            "secret_write_times",
            lambda ns, names: {"creds": datetime(2026, 7, 30, 13, 0)},
        )
        monkeypatch.setattr(
            deploy, "workload_pod_starts", lambda d, ns: [datetime(2026, 7, 29, 6, 0)]
        )
        stale = deploy.find_stale_secret_consumers(
            self.plan, {("CronJob", "publisher"): doc}, "ns"
        )
        assert stale == {}

    def test_missing_secret_is_not_flagged(self, monkeypatch):
        """First install, or a Secret the operator has not created yet."""
        doc = workload("Deployment", "trace-exporter", secrets=["creds"])
        monkeypatch.setattr(deploy, "secret_write_times", lambda ns, names: {})
        monkeypatch.setattr(
            deploy, "workload_pod_starts", lambda d, ns: [datetime(2026, 7, 29, 6, 0)]
        )
        assert (
            deploy.find_stale_secret_consumers(
                self.plan, {("Deployment", "trace-exporter"): doc}, "ns"
            )
            == {}
        )

    def test_no_running_pods_is_not_flagged(self, monkeypatch):
        doc = workload("Deployment", "trace-exporter", secrets=["creds"])
        monkeypatch.setattr(
            deploy,
            "secret_write_times",
            lambda ns, names: {"creds": datetime(2026, 7, 30, 13, 0)},
        )
        monkeypatch.setattr(deploy, "workload_pod_starts", lambda d, ns: [])
        assert (
            deploy.find_stale_secret_consumers(
                self.plan, {("Deployment", "trace-exporter"): doc}, "ns"
            )
            == {}
        )

    def test_records_timestamps_for_the_operator(self, monkeypatch):
        self.check(
            monkeypatch,
            written=datetime(2026, 7, 30, 13, 23, 30),
            pod_started=datetime(2026, 7, 29, 6, 54, 20),
        )
        detail = self.plan.stale_secret_detail[("Deployment", "trace-exporter")][0]
        assert "2026-07-30T13:23:30Z" in detail
        assert "2026-07-29T06:54:20Z" in detail


class TestTimestampParsing:
    def test_parses_plain_rfc3339(self):
        assert deploy.parse_k8s_time("2026-07-30T13:23:30Z") == datetime(
            2026, 7, 30, 13, 23, 30
        )

    def test_parses_fractional_seconds(self):
        assert deploy.parse_k8s_time("2026-07-30T13:23:30.123456Z") == datetime(
            2026, 7, 30, 13, 23, 30, 123456
        )

    def test_rejects_non_utc_and_junk(self):
        assert deploy.parse_k8s_time("2026-07-30T13:23:30+02:00") is None
        assert deploy.parse_k8s_time("") is None
        assert deploy.parse_k8s_time("not a time") is None


class TestManifestParsing:
    def test_parses_concatenated_json_documents(self):
        text = '{"kind": "A", "metadata": {"name": "x"}}\n{"kind": "B", "metadata": {"name": "y"}}'
        docs = deploy.parse_json_stream(text)
        assert [d["kind"] for d in docs] == ["A", "B"]

    def test_flattens_list_documents(self):
        text = '{"kind": "List", "items": [{"kind": "A"}, {"kind": "B"}]}'
        assert [d["kind"] for d in deploy.parse_json_stream(text)] == ["A", "B"]

    def test_handles_crlf_line_endings(self):
        """Windows subprocess output arrives with \\r\\n."""
        text = '{"kind": "A"}\r\n{"kind": "B"}\r\n'
        assert len(deploy.parse_json_stream(text)) == 2

    def test_config_refs_finds_every_reference_style(self):
        template = {
            "spec": {
                "volumes": [
                    {"name": "v", "configMap": {"name": "cm-volume"}},
                    {"name": "s", "secret": {"secretName": "secret-volume"}},
                    {
                        "name": "p",
                        "projected": {
                            "sources": [
                                {"configMap": {"name": "cm-projected"}},
                                {"secret": {"name": "secret-projected"}},
                            ]
                        },
                    },
                ],
                "initContainers": [
                    {"name": "i", "envFrom": [{"configMapRef": {"name": "cm-envfrom"}}]}
                ],
                "containers": [
                    {
                        "name": "c",
                        "envFrom": [{"secretRef": {"name": "secret-envfrom"}}],
                        "env": [
                            {
                                "name": "A",
                                "valueFrom": {"configMapKeyRef": {"name": "cm-key"}},
                            },
                            {
                                "name": "B",
                                "valueFrom": {"secretKeyRef": {"name": "secret-key"}},
                            },
                            {"name": "C", "value": "literal"},
                        ],
                    }
                ],
            }
        }
        configmaps, secrets = deploy.config_refs(template)
        assert configmaps == {"cm-volume", "cm-projected", "cm-envfrom", "cm-key"}
        assert secrets == {
            "secret-volume",
            "secret-projected",
            "secret-envfrom",
            "secret-key",
        }

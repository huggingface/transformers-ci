#!/usr/bin/env python3
"""Deploy the transformers-ci Helm chart."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CHART_DIR = ROOT_DIR / "deploy" / "helm"
DEPLOYMENTS = ("grafana", "otelcol", "trace-exporter")
STATEFULSETS = ("tempo", "prometheus")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy/scripts/deploy.sh",
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
    parser.add_argument("--context", help="Require this kubectl context before deploying")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render manifests without changing the cluster",
    )
    return parser


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

    if args.dry_run:
        run(
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
        return 0

    namespace_exists = run(
        ["kubectl", "get", "namespace", args.namespace], check=False, quiet=True
    ).returncode == 0
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
            "10m",
        ]
    )

    for deployment in DEPLOYMENTS:
        run(
            [
                "kubectl",
                "rollout",
                "status",
                f"deployment/{deployment}",
                "-n",
                args.namespace,
                "--timeout=10m",
            ]
        )
    for statefulset in STATEFULSETS:
        run(
            [
                "kubectl",
                "rollout",
                "status",
                f"statefulset/{statefulset}",
                "-n",
                args.namespace,
                "--timeout=10m",
            ]
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Read logs from the newest running transformers-ci component pod."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys


LOG_START_RE = re.compile(r"^[0-9-]+ [0-9:,]+ (INFO|WARNING|ERROR|DEBUG) ")
ERROR_START_RE = re.compile(r"^[0-9-]+ [0-9:,]+ ERROR ")


def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing required command: {name}")


def output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def run_stream(args: list[str], grep_pattern: str = "") -> int:
    proc = subprocess.Popen(args, text=True, stdout=subprocess.PIPE)
    assert proc.stdout is not None

    regex = re.compile(grep_pattern, re.IGNORECASE) if grep_pattern else None
    matched = False
    try:
        for line in proc.stdout:
            if regex is None or regex.search(line):
                matched = True
                print(line, end="")
    except BrokenPipeError:
        proc.stdout.close()
        return proc.wait()
    returncode = proc.wait()
    if returncode == 0 and regex is not None and not matched:
        return 1
    return returncode


def newest_running_pod(namespace: str, component: str) -> str:
    pods = output(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app.kubernetes.io/component={component}",
            "--field-selector=status.phase=Running",
            "--sort-by=.metadata.creationTimestamp",
            "-o",
            'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}',
        ]
    )
    return pods.splitlines()[-1] if pods else ""


def last_error_block(lines: list[str]) -> str:
    last = ""
    block: list[str] = []
    in_block = False

    def flush() -> None:
        nonlocal last, block, in_block
        if in_block and block:
            last = "".join(block)
        block = []
        in_block = False

    for line in lines:
        if ERROR_START_RE.match(line):
            flush()
            in_block = True
            block = [line]
            continue

        if line.startswith("Traceback (most recent call last):"):
            if not in_block:
                in_block = True
                block = [line]
            else:
                block.append(line)
            continue

        if in_block and (LOG_START_RE.match(line) or line.startswith("INFO:")):
            flush()

        if in_block:
            block.append(line)

    flush()
    return last


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deploy/scripts/logs.py")
    parser.add_argument(
        "-n",
        "--namespace",
        default="transformers-ci",
        help="Kubernetes namespace (default: transformers-ci)",
    )
    parser.add_argument(
        "-c",
        "--component",
        default="grafana",
        help=(
            "Component to read (default: grafana). One of: grafana, otelcol, "
            "trace-exporter, tempo, prometheus, ci-data-publisher"
        ),
    )
    parser.add_argument(
        "--since", default="2h", help="Log window, e.g. 30m, 2h (default: 2h)"
    )
    parser.add_argument("-f", "--follow", action="store_true", help="Follow logs")
    parser.add_argument(
        "--grep", dest="grep_pattern", help="Filter logs with case-insensitive regex"
    )
    parser.add_argument(
        "--last-error",
        action="store_true",
        help="Print the last ERROR/Traceback block from recent logs",
    )
    parser.add_argument(
        "--context", help="Require this kubectl context before reading logs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    require_cmd("kubectl")

    try:
        current_context = output(["kubectl", "config", "current-context"])
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    if args.context and current_context != args.context:
        raise SystemExit(
            f"refusing to read logs from context '{current_context}' (expected '{args.context}')"
        )

    try:
        pod = newest_running_pod(args.namespace, args.component)
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    if not pod:
        print(
            "no running pod found in namespace "
            f"'{args.namespace}' with label app.kubernetes.io/component={args.component}",
            file=sys.stderr,
        )
        return 1

    pod_started = output(
        [
            "kubectl",
            "get",
            "pod",
            pod,
            "-n",
            args.namespace,
            "-o",
            "jsonpath={.status.startTime}",
        ]
    )

    print(f"Context: {current_context}", file=sys.stderr)
    print(f"Namespace: {args.namespace}", file=sys.stderr)
    print(f"Component: {args.component}", file=sys.stderr)
    print(f"Pod: {pod}", file=sys.stderr)
    print(f"Pod started: {pod_started}", file=sys.stderr)
    print(f"Since: {args.since}", file=sys.stderr)

    kubectl_args = [
        "kubectl",
        "logs",
        "-n",
        args.namespace,
        pod,
        f"--since={args.since}",
    ]
    if args.follow:
        kubectl_args.append("-f")

    if args.last_error:
        if args.follow:
            print("--last-error cannot be combined with --follow", file=sys.stderr)
            return 2
        try:
            logs = subprocess.check_output(kubectl_args, text=True).splitlines(
                keepends=True
            )
        except subprocess.CalledProcessError as exc:
            return exc.returncode
        block = last_error_block(logs)
        if block:
            print(block, end="")
            return 0
        print(
            f"no ERROR/Traceback block found for component {args.component} in the last {args.since}",
            file=sys.stderr,
        )
        print(
            "note: this only covers logs retained for the current pod, "
            f"which started at {pod_started}",
            file=sys.stderr,
        )
        return 1

    try:
        return run_stream(kubectl_args, args.grep_pattern or "")
    except re.error as exc:
        print(f"invalid --grep pattern: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

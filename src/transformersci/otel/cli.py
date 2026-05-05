#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
from collections.abc import Mapping, Sequence
from urllib.parse import urlparse

DEFAULT_SERVICE_NAME = "transformers-tests"
DEFAULT_LOCAL_SUITE = "local_pytest"
LOCAL_PROVIDER = "local"
OTEL_PING_TIMEOUT_SECONDS = 2.0


def has_otel_endpoint(env: Mapping[str, str]) -> bool:
    return bool(
        env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def resolve_otel_endpoint(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    for key in ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        endpoint = env.get(key)
        if endpoint:
            return key, endpoint

    return None, None


def endpoint_target(endpoint: str, env: Mapping[str, str]) -> tuple[str, int] | None:
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    if parsed.hostname is None:
        return None

    port = parsed.port
    if port is None:
        if parsed.scheme == "https":
            port = 443
        else:
            protocol = env.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
            port = 4318 if protocol == "http/protobuf" else 4317

    return parsed.hostname, port


def ping_server(
    env: Mapping[str, str], *, timeout_seconds: float = OTEL_PING_TIMEOUT_SECONDS
) -> bool:
    endpoint_source, endpoint = resolve_otel_endpoint(env)
    if endpoint is None:
        print("OTEL PING SKIPPED endpoint is not configured", flush=True)
        return False

    target = endpoint_target(endpoint, env)
    if target is None:
        print(
            f"OTEL PING FAILED source={endpoint_source} endpoint={endpoint} error=unable to parse host/port",
            flush=True,
        )
        return False

    host, port = target
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            print(
                f"OTEL PING OK source={endpoint_source} endpoint={endpoint} target={host}:{port} timeout={timeout_seconds}s",
                flush=True,
            )
            return True
    except OSError as error:
        print(
            f"OTEL PING FAILED source={endpoint_source} endpoint={endpoint} target={host}:{port} timeout={timeout_seconds}s error={error}",
            flush=True,
        )
        return False


def detect_provider(env: Mapping[str, str]) -> str:
    if env.get("GITHUB_ACTIONS"):
        return "github_actions"
    if env.get("CIRCLECI") or env.get("CIRCLE_WORKFLOW_ID"):
        return "circleci"
    return LOCAL_PROVIDER


def default_suite(env: Mapping[str, str], provider: str) -> str:
    if provider == "github_actions":
        return env.get("GITHUB_JOB", "github_actions_pytest")
    if provider == "circleci":
        return env.get("CIRCLE_JOB", "circleci_pytest")
    return DEFAULT_LOCAL_SUITE


def append_resource_attributes(
    existing: str | None, new_attributes: Sequence[str]
) -> str:
    segments = [segment for segment in [existing, ",".join(new_attributes)] if segment]
    return ",".join(segments)


def read_github_event(env: Mapping[str, str]) -> dict | None:
    event_path = env.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None

    try:
        with open(event_path, encoding="utf-8") as event_file:
            return json.load(event_file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def github_pr_number(env: Mapping[str, str]) -> str | None:
    event = read_github_event(env)
    if event is None:
        return None

    pull_request = event.get("pull_request")
    if isinstance(pull_request, dict) and pull_request.get("number") is not None:
        return str(pull_request["number"])

    issue = event.get("issue")
    if (
        isinstance(issue, dict)
        and issue.get("pull_request")
        and issue.get("number") is not None
    ):
        return str(issue["number"])

    return None


def circleci_pr_number(env: Mapping[str, str]) -> str | None:
    pull_request = env.get("CIRCLE_PULL_REQUEST", "")
    if pull_request:
        return pull_request.rstrip("/").split("/")[-1] or None

    pull_requests = env.get("CIRCLE_PULL_REQUESTS", "")
    if pull_requests:
        first_pull_request = pull_requests.split(",")[0].strip()
        if first_pull_request:
            return first_pull_request.rstrip("/").split("/")[-1] or None

    branch = env.get("CIRCLE_BRANCH", "")
    if branch.startswith("pull/"):
        return branch.split("/")[1]

    return None


def suite_run_id(env: Mapping[str, str], provider: str, suite: str) -> str | None:
    if provider == "github_actions":
        run_id = env.get("GITHUB_RUN_ID")
        if not run_id:
            return None

        run_attempt = env.get("GITHUB_RUN_ATTEMPT")
        segments = [run_id, suite]
        if run_attempt:
            segments.append(run_attempt)
        return ":".join(segments)

    if provider == "circleci":
        build_num = env.get("CIRCLE_BUILD_NUM")
        if build_num:
            return build_num

        workflow_id = env.get("CIRCLE_WORKFLOW_ID")
        if workflow_id:
            return f"{workflow_id}:{suite}"

    return None


def build_resource_attributes(
    env: Mapping[str, str],
    provider: str,
    suite: str,
    pr_number: str | None,
) -> list[str]:
    if provider == "github_actions":
        attributes = [
            "deployment.environment=ci",
            "transformers.test.provider=github_actions",
            f"transformers.test.suite={suite}",
            f"cicd.pipeline.run.id={env.get('GITHUB_RUN_ID', 'unknown')}",
            f"cicd.pipeline.task.name={env.get('GITHUB_JOB', 'unknown')}",
            f"vcs.ref.head.name={env.get('GITHUB_REF_NAME', 'unknown')}",
            f"vcs.ref.head.revision={env.get('GITHUB_SHA', 'unknown')}",
            "vcs.ref.type=branch",
            "vcs.provider.name=github",
        ]
        resolved_suite_run_id = suite_run_id(env, provider, suite)
        if pr_number:
            attributes.append(f"vcs.change.id={pr_number}")
        if resolved_suite_run_id is not None:
            attributes.append(f"transformers.test.suite.run={resolved_suite_run_id}")
        return attributes

    if provider == "circleci":
        attributes = [
            "deployment.environment=ci",
            "transformers.test.provider=circleci",
            f"transformers.test.suite={suite}",
            f"cicd.pipeline.run.id={env.get('CIRCLE_WORKFLOW_ID', 'unknown')}",
            f"cicd.pipeline.task.name={env.get('CIRCLE_JOB', 'unknown')}",
            f"vcs.ref.head.name={env.get('CIRCLE_BRANCH', 'unknown')}",
            f"vcs.ref.head.revision={env.get('CIRCLE_SHA1', 'unknown')}",
            "vcs.ref.type=branch",
            "vcs.provider.name=github",
        ]
        resolved_suite_run_id = suite_run_id(env, provider, suite)
        if pr_number:
            attributes.append(f"vcs.change.id={pr_number}")
        if resolved_suite_run_id is not None:
            attributes.append(f"transformers.test.suite.run={resolved_suite_run_id}")
        return attributes

    attributes = [
        "deployment.environment=local",
        f"transformers.test.provider={LOCAL_PROVIDER}",
        f"transformers.test.suite={suite}",
    ]
    if pr_number:
        attributes.append(f"vcs.change.id={pr_number}")
    return attributes


def is_pytest_command(command: Sequence[str]) -> bool:
    return any(token == "pytest" or token.endswith("/pytest") for token in command)


def traceparent_from_command(command: Sequence[str]) -> str | None:
    for index, token in enumerate(command):
        if token == "--trace-parent" and index + 1 < len(command):
            return command[index + 1]
        if token.startswith("--trace-parent="):
            return token.partition("=")[2]
    return None


def trace_id_from_traceparent(traceparent: str | None) -> str | None:
    if traceparent is None:
        return None

    parts = traceparent.strip().split("-")
    if len(parts) != 4:
        return None

    version, trace_id, span_id, trace_flags = parts
    if (
        len(version) != 2
        or len(trace_id) != 32
        or len(span_id) != 16
        or len(trace_flags) != 2
    ):
        return None

    try:
        int(version, 16)
        int(trace_id, 16)
        int(span_id, 16)
        int(trace_flags, 16)
    except ValueError:
        return None

    if int(trace_id, 16) == 0 or int(span_id, 16) == 0:
        return None

    return trace_id.lower()


def generate_traceparent() -> str:
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def configure_trace_context(
    env: Mapping[str, str],
    command: Sequence[str],
    *,
    export_traces: bool,
) -> tuple[dict[str, str], str | None]:
    updated_env = dict(env)

    if not export_traces or not is_pytest_command(command):
        return updated_env, None

    traceparent = traceparent_from_command(command) or env.get("TRACEPARENT")
    if traceparent is None:
        traceparent = generate_traceparent()
        updated_env["TRACEPARENT"] = traceparent

    return updated_env, trace_id_from_traceparent(traceparent)


def emit_trace_log(
    phase: str,
    trace_id: str,
    env: Mapping[str, str],
    command: Sequence[str],
    *,
    exit_code: int | None = None,
) -> None:
    details = [
        f"trace_id={trace_id}",
        f"service={env.get('OTEL_SERVICE_NAME', DEFAULT_SERVICE_NAME)}",
        f"suite={env.get('TRANSFORMERS_TEST_OTEL_SUITE', DEFAULT_LOCAL_SUITE)}",
    ]
    if exit_code is not None:
        details.append(f"exit_code={exit_code}")
    if phase == "start":
        details.append(f"command={' '.join(command)}")
    print(f"OTEL TRACE {phase.upper()} " + " ".join(details), flush=True)


def prepare_environment(
    env: Mapping[str, str],
    *,
    suite: str | None = None,
    service_name: str | None = None,
    force_export_traces: bool = False,
) -> tuple[dict[str, str], bool]:
    should_export_traces = force_export_traces or has_otel_endpoint(env)
    updated_env = dict(env)

    if not should_export_traces:
        return updated_env, False

    provider = detect_provider(env)
    resolved_suite = (
        suite or env.get("TRANSFORMERS_TEST_OTEL_SUITE") or default_suite(env, provider)
    )
    resolved_service_name = (
        service_name or env.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    )
    resolved_pr = env.get("TRANSFORMERS_TEST_OTEL_PR") or None
    if resolved_pr is None:
        if provider == "github_actions":
            resolved_pr = github_pr_number(env)
        elif provider == "circleci":
            resolved_pr = circleci_pr_number(env)

    updated_env["TRANSFORMERS_TEST_OTEL_SUITE"] = resolved_suite
    updated_env["OTEL_SERVICE_NAME"] = resolved_service_name
    if resolved_pr:
        updated_env["TRANSFORMERS_TEST_OTEL_PR"] = resolved_pr
    updated_env.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    attributes = build_resource_attributes(env, provider, resolved_suite, resolved_pr)
    updated_env["OTEL_RESOURCE_ATTRIBUTES"] = append_resource_attributes(
        env.get("OTEL_RESOURCE_ATTRIBUTES"),
        attributes,
    )

    return updated_env, True


def augment_pytest_command(command: Sequence[str], *, export_traces: bool) -> list[str]:
    augmented_command = list(command)
    if not export_traces or "--export-traces" in augmented_command:
        return augmented_command

    if any(
        token == "pytest" or token.endswith("/pytest") for token in augmented_command
    ):
        augmented_command.append("--export-traces")

    return augmented_command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest with OpenTelemetry configured for CI or local testing."
    )
    parser.add_argument(
        "--suite", help="Override the test suite attribute (transformers.test.suite)."
    )
    parser.add_argument(
        "--service-name", help="Override the OpenTelemetry service.name."
    )
    parser.add_argument(
        "--ping-server",
        action="store_true",
        help="Best-effort TCP connectivity check for the configured OTLP endpoint. Prints a log line and never fails the command.",
    )
    parser.add_argument(
        "--force-export-traces",
        action="store_true",
        help="Enable pytest trace exporting even without an explicit OTLP endpoint env var.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved OpenTelemetry configuration as JSON before running the command.",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to execute after '--'."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]

    env, export_traces = prepare_environment(
        os.environ,
        suite=args.suite,
        service_name=args.service_name,
        force_export_traces=args.force_export_traces,
    )
    command = augment_pytest_command(command, export_traces=export_traces)
    env, trace_id = configure_trace_context(env, command, export_traces=export_traces)

    if args.print_config:
        print(
            json.dumps(
                {
                    "export_traces": export_traces,
                    "suite": env.get("TRANSFORMERS_TEST_OTEL_SUITE"),
                    "service_name": env.get("OTEL_SERVICE_NAME"),
                    "resource_attributes": env.get("OTEL_RESOURCE_ATTRIBUTES"),
                    "trace_id": trace_id,
                    "command": command,
                },
                sort_keys=True,
            )
        )

    if args.ping_server:
        ping_server(env)

    if not command:
        if args.print_config or args.ping_server:
            return 0
        raise SystemExit("A command is required after '--'.")

    if trace_id is not None:
        emit_trace_log("start", trace_id, env, command)

    exit_code = subprocess.run(command, env=env, check=False).returncode

    if trace_id is not None:
        emit_trace_log("end", trace_id, env, command, exit_code=exit_code)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

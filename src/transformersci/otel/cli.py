#!/usr/bin/env python3
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
"""``transformersci-otel`` — run pytest with OpenTelemetry trace export wired up.

This is the producer-side entry point. It wraps a pytest invocation so that each
CI run emits a trace whose spans carry enough metadata (provider, repo, PR,
run/job IDs) for the dashboards to group and label runs downstream.

Organization (top to bottom):

- OTLP endpoint/protocol/transport resolution — read the standard ``OTEL_*``
  env vars, normalize them, and pick the right exporter, with an optional
  reachability ``ping_server`` preflight.
- CI-provider detection and metadata — GitHub Actions and CircleCI helpers that
  dig the repository, PR number/URL, and workflow/job run IDs out of the
  provider's environment, plus :func:`build_resource_attributes` which folds
  them into OTel resource attributes.
- Trace context — generate or inherit a W3C ``traceparent`` so a run's spans
  share one trace id (:func:`configure_trace_context`).
- Command assembly — :func:`prepare_environment` and
  :func:`augment_pytest_command` build the child environment and argv, and
  :func:`main` parses args and execs pytest.
"""

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
DEFAULT_LOCAL_JOB = "local_pytest"
LOCAL_PROVIDER = "local"
OTEL_PING_TIMEOUT_SECONDS = 2.0
# Read by the pytest plugin (resource_plugin) to attach a SECOND span processor
# so every span is mirrored to a staging backend on top of the primary export.
STAGING_ENDPOINT_ENV = "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT"
# Bearer header for the staging mirror. Kept separate from the primary's
# OTEL_EXPORTER_OTLP_HEADERS so staging can authenticate with its own token.
STAGING_HEADERS_ENV = "TRANSFORMERS_TEST_OTEL_STAGING_HEADERS"
STAGING_TOKEN_ENV = "TRANSFORMERS_TEST_OTEL_STAGING_TOKEN"
# OTLP transport for the staging mirror. Falls back to the primary protocol
# (OTEL_EXPORTER_OTLP_PROTOCOL) when unset, so staging can use a different
# transport than prod (e.g. grpc to a stage box that only speaks gRPC).
STAGING_PROTOCOL_ENV = "TRANSFORMERS_TEST_OTEL_STAGING_PROTOCOL"
OTEL_TRACES_EXPORTER_BY_PROTOCOL = {
    "grpc": "otlp_proto_grpc",
    "http/protobuf": "otlp_proto_http",
}
OTEL_PROTOCOL_BY_TRACES_EXPORTER = {
    exporter: protocol
    for protocol, exporter in OTEL_TRACES_EXPORTER_BY_PROTOCOL.items()
}


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


def normalize_otel_protocol(protocol: str | None) -> str | None:
    if not protocol:
        return None

    if protocol in {"http", "https", "http/protobuf"}:
        return "http/protobuf"

    return protocol


def normalize_protocol_override(protocol: str | None) -> str | None:
    if not protocol:
        return None

    if protocol == "http":
        return "http/protobuf"

    return protocol


def resolve_otel_protocol(env: Mapping[str, str]) -> str:
    configured_protocol = normalize_otel_protocol(
        env.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    )
    if configured_protocol is not None:
        return configured_protocol

    _, endpoint = resolve_otel_endpoint(env)
    if endpoint is not None:
        parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
        if parsed.scheme in {"http", "https"}:
            return "http/protobuf"

    return "grpc"


def resolve_otel_transport(
    env: Mapping[str, str], *, protocol_override: str | None = None
) -> tuple[str, str]:
    resolved_override = normalize_protocol_override(protocol_override)
    if resolved_override is not None:
        return OTEL_TRACES_EXPORTER_BY_PROTOCOL[resolved_override], resolved_override

    configured_exporter = env.get("OTEL_TRACES_EXPORTER")
    if configured_exporter in OTEL_PROTOCOL_BY_TRACES_EXPORTER:
        return (
            configured_exporter,
            OTEL_PROTOCOL_BY_TRACES_EXPORTER[configured_exporter],
        )

    resolved_protocol = resolve_otel_protocol(env)
    if configured_exporter:
        return configured_exporter, resolved_protocol

    traces_exporter = OTEL_TRACES_EXPORTER_BY_PROTOCOL.get(resolved_protocol, "otlp")
    return traces_exporter, resolved_protocol


def endpoint_target(endpoint: str, env: Mapping[str, str]) -> tuple[str, int] | None:
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    if parsed.hostname is None:
        return None

    port = parsed.port
    if port is None:
        if parsed.scheme == "https":
            port = 443
        else:
            _, protocol = resolve_otel_transport(env)
            port = 4318 if protocol == "http/protobuf" else 4317

    return parsed.hostname, port


def ping_server(
    env: Mapping[str, str],
    *,
    timeout_seconds: float = OTEL_PING_TIMEOUT_SECONDS,
    endpoint: str | None = None,
    endpoint_source: str | None = None,
) -> bool:
    if endpoint is None:
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


def default_job(env: Mapping[str, str], provider: str) -> str:
    if provider == "github_actions":
        return env.get("GITHUB_JOB", "github_actions_pytest")
    if provider == "circleci":
        return env.get("CIRCLE_JOB", "circleci_pytest")
    return DEFAULT_LOCAL_JOB


def append_resource_attributes(
    existing: str | None, new_attributes: Sequence[str]
) -> str:
    segments = [segment for segment in [existing, ",".join(new_attributes)] if segment]
    return ",".join(segments)


def bearer_auth_header(token: str) -> str:
    return f"Authorization=Bearer {token}"


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


def github_repository(env: Mapping[str, str]) -> str | None:
    event = read_github_event(env)
    if event is not None:
        repository = event.get("repository")
        if isinstance(repository, dict):
            full_name = repository.get("full_name")
            if isinstance(full_name, str) and full_name:
                return full_name

        pull_request = event.get("pull_request")
        if isinstance(pull_request, dict):
            base = pull_request.get("base")
            if isinstance(base, dict):
                repo = base.get("repo")
                if isinstance(repo, dict):
                    full_name = repo.get("full_name")
                    if isinstance(full_name, str) and full_name:
                        return full_name

    repository_name = env.get("GITHUB_REPOSITORY", "")
    return repository_name or None


def github_pr_url(env: Mapping[str, str]) -> str | None:
    event = read_github_event(env)
    if event is None:
        return None

    pull_request = event.get("pull_request")
    if isinstance(pull_request, dict):
        html_url = pull_request.get("html_url")
        if isinstance(html_url, str) and html_url:
            return html_url

    issue = event.get("issue")
    if (
        isinstance(issue, dict)
        and issue.get("pull_request")
        and isinstance(issue.get("html_url"), str)
        and issue["html_url"]
    ):
        return issue["html_url"]

    return None


def circleci_repository(env: Mapping[str, str]) -> str | None:
    owner = env.get("CIRCLE_PROJECT_USERNAME", "").strip()
    repo = env.get("CIRCLE_PROJECT_REPONAME", "").strip()
    if owner and repo:
        return f"{owner}/{repo}"
    return None


def circleci_pr_url(env: Mapping[str, str]) -> str | None:
    pull_request = env.get("CIRCLE_PULL_REQUEST", "").strip()
    if pull_request:
        return pull_request

    pull_requests = env.get("CIRCLE_PULL_REQUESTS", "")
    if pull_requests:
        first_pull_request = pull_requests.split(",")[0].strip()
        if first_pull_request:
            return first_pull_request

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


def workflow_run_id(env: Mapping[str, str], provider: str) -> str | None:
    explicit_run_id = env.get("TRANSFORMERS_TEST_OTEL_RUN_ID")
    if explicit_run_id:
        return explicit_run_id

    if provider == "github_actions":
        run_id = env.get("GITHUB_RUN_ID")
        if not run_id:
            return None

        run_attempt = env.get("GITHUB_RUN_ATTEMPT")
        if run_attempt:
            return f"{run_id}:{run_attempt}"
        return run_id

    if provider == "circleci":
        workflow_id = env.get("CIRCLE_WORKFLOW_ID")
        if workflow_id:
            return workflow_id

        build_num = env.get("CIRCLE_BUILD_NUM")
        if build_num:
            return build_num

    return None


def job_run_id(env: Mapping[str, str], provider: str, job: str) -> str | None:
    run_id = workflow_run_id(env, provider)
    if provider == "github_actions":
        if run_id is None:
            return None
        return f"{run_id}:{job}"

    if provider == "circleci":
        build_num = env.get("CIRCLE_BUILD_NUM")
        if build_num:
            return build_num

        if run_id is not None:
            return f"{run_id}:{job}"

    return None


def build_resource_attributes(
    env: Mapping[str, str],
    provider: str,
    job: str,
    pr_number: str | None,
) -> list[str]:
    repository_name = env.get("TRANSFORMERS_TEST_OTEL_REPOSITORY")
    pr_url = env.get("TRANSFORMERS_TEST_OTEL_PR_URL")
    if not repository_name:
        if provider == "github_actions":
            repository_name = github_repository(env)
        elif provider == "circleci":
            repository_name = circleci_repository(env)
    if not pr_url:
        if provider == "github_actions":
            pr_url = github_pr_url(env)
        elif provider == "circleci":
            pr_url = circleci_pr_url(env)

    if provider == "github_actions":
        resolved_run_id = workflow_run_id(env, provider)
        attributes = [
            "deployment.environment=ci",
            "transformers.test.provider=github_actions",
            f"transformers.test.job={job}",
            f"cicd.pipeline.run.id={resolved_run_id or env.get('GITHUB_RUN_ID', 'unknown')}",
            f"cicd.pipeline.task.name={env.get('GITHUB_JOB', 'unknown')}",
            f"vcs.ref.head.name={env.get('GITHUB_REF_NAME', 'unknown')}",
            f"vcs.ref.head.revision={env.get('GITHUB_SHA', 'unknown')}",
            "vcs.ref.type=branch",
            "vcs.provider.name=github",
        ]
        resolved_job_run_id = job_run_id(env, provider, job)
        if pr_number:
            attributes.append(f"vcs.change.id={pr_number}")
        if pr_url:
            attributes.append(f"vcs.change.url={pr_url}")
        if repository_name:
            attributes.append(f"vcs.repository.name={repository_name}")
        if resolved_run_id is not None:
            attributes.append(f"transformers.test.run.id={resolved_run_id}")
        if resolved_job_run_id is not None:
            attributes.append(f"transformers.test.job.run={resolved_job_run_id}")
        return attributes

    if provider == "circleci":
        resolved_run_id = workflow_run_id(env, provider)
        attributes = [
            "deployment.environment=ci",
            "transformers.test.provider=circleci",
            f"transformers.test.job={job}",
            f"cicd.pipeline.run.id={resolved_run_id or env.get('CIRCLE_WORKFLOW_ID', 'unknown')}",
            f"cicd.pipeline.task.name={env.get('CIRCLE_JOB', 'unknown')}",
            f"vcs.ref.head.name={env.get('CIRCLE_BRANCH', 'unknown')}",
            f"vcs.ref.head.revision={env.get('CIRCLE_SHA1', 'unknown')}",
            "vcs.ref.type=branch",
            "vcs.provider.name=github",
        ]
        resolved_job_run_id = job_run_id(env, provider, job)
        if pr_number:
            attributes.append(f"vcs.change.id={pr_number}")
        if pr_url:
            attributes.append(f"vcs.change.url={pr_url}")
        if repository_name:
            attributes.append(f"vcs.repository.name={repository_name}")
        if resolved_run_id is not None:
            attributes.append(f"transformers.test.run.id={resolved_run_id}")
        if resolved_job_run_id is not None:
            attributes.append(f"transformers.test.job.run={resolved_job_run_id}")
        return attributes

    attributes = [
        "deployment.environment=local",
        f"transformers.test.provider={LOCAL_PROVIDER}",
        f"transformers.test.job={job}",
    ]
    resolved_run_id = workflow_run_id(env, LOCAL_PROVIDER)
    if resolved_run_id is not None:
        attributes.append(f"transformers.test.run.id={resolved_run_id}")
        attributes.append(f"transformers.test.job.run={resolved_run_id}:{job}")
    if pr_number:
        attributes.append(f"vcs.change.id={pr_number}")
    if pr_url:
        attributes.append(f"vcs.change.url={pr_url}")
    if repository_name:
        attributes.append(f"vcs.repository.name={repository_name}")
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
        f"job={env.get('TRANSFORMERS_TEST_OTEL_JOB', DEFAULT_LOCAL_JOB)}",
    ]
    if exit_code is not None:
        details.append(f"exit_code={exit_code}")
    if phase == "start":
        details.append(f"command={' '.join(command)}")
    print(f"OTEL TRACE {phase.upper()} " + " ".join(details), flush=True)


def prepare_environment(
    env: Mapping[str, str],
    *,
    job: str | None = None,
    service_name: str | None = None,
    force_export_traces: bool = False,
    protocol: str | None = None,
    otlp_endpoint: str | None = None,
    staging_endpoint: str | None = None,
    staging_protocol: str | None = None,
    token: str | None = None,
    staging_token: str | None = None,
    pr: str | None = None,
) -> tuple[dict[str, str], bool]:
    updated_env = dict(env)
    if otlp_endpoint:
        updated_env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
        updated_env.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)
    if staging_endpoint:
        # The pytest plugin reads this to add a second span processor pointing
        # at the staging backend, mirroring every span the primary exporter
        # sends.
        updated_env[STAGING_ENDPOINT_ENV] = staging_endpoint
    resolved_staging_protocol = normalize_protocol_override(staging_protocol)
    if resolved_staging_protocol:
        # Stored normalized ("http/protobuf"/"grpc") so the plugin can pick the
        # matching exporter without re-parsing. Independent of the primary so
        # staging can speak a different transport.
        updated_env[STAGING_PROTOCOL_ENV] = resolved_staging_protocol
    resolved_token = token or updated_env.get("OTEL_EXPORTER_OTLP_TOKEN")
    if resolved_token:
        headers = bearer_auth_header(resolved_token)
        updated_env["OTEL_EXPORTER_OTLP_HEADERS"] = headers
        updated_env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = headers
    resolved_staging_token = staging_token or updated_env.get(STAGING_TOKEN_ENV)
    if resolved_staging_token:
        # Staging gets its own bearer header so it can authenticate
        # independently of the primary backend's token.
        updated_env[STAGING_HEADERS_ENV] = bearer_auth_header(resolved_staging_token)

    should_export_traces = force_export_traces or has_otel_endpoint(updated_env)

    if not should_export_traces:
        return updated_env, False

    provider = detect_provider(env)
    resolved_job = (
        job
        or env.get("TRANSFORMERS_TEST_OTEL_JOB")
        or env.get("TRANSFORMERS_TEST_OTEL_SUITE")
        or default_job(env, provider)
    )
    resolved_service_name = (
        service_name or env.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    )
    resolved_pr = pr or env.get("TRANSFORMERS_TEST_OTEL_PR") or None
    if resolved_pr is None:
        if provider == "github_actions":
            resolved_pr = github_pr_number(env)
        elif provider == "circleci":
            resolved_pr = circleci_pr_number(env)
    resolved_run_id = workflow_run_id(updated_env, provider)

    updated_env["TRANSFORMERS_TEST_OTEL_JOB"] = resolved_job
    updated_env["OTEL_SERVICE_NAME"] = resolved_service_name
    if resolved_pr:
        updated_env["TRANSFORMERS_TEST_OTEL_PR"] = resolved_pr
    if resolved_run_id:
        updated_env["TRANSFORMERS_TEST_OTEL_RUN_ID"] = resolved_run_id
    traces_exporter, otel_protocol = resolve_otel_transport(
        updated_env, protocol_override=protocol
    )
    updated_env["OTEL_TRACES_EXPORTER"] = traces_exporter
    updated_env["OTEL_EXPORTER_OTLP_PROTOCOL"] = otel_protocol
    attributes = build_resource_attributes(
        updated_env, provider, resolved_job, resolved_pr
    )
    updated_env["OTEL_RESOURCE_ATTRIBUTES"] = append_resource_attributes(
        updated_env.get("OTEL_RESOURCE_ATTRIBUTES"),
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
        "--job",
        "--suite",
        dest="job",
        help="Override the test job attribute (transformers.test.job). --suite is accepted as a back-compat alias.",
    )
    parser.add_argument(
        "--service-name", help="Override the OpenTelemetry service.name."
    )
    parser.add_argument(
        "--protocol",
        choices=("http", "grpc"),
        help="Override the OTLP transport and set OTEL_TRACES_EXPORTER plus OTEL_EXPORTER_OTLP_PROTOCOL consistently.",
    )
    parser.add_argument(
        "--token",
        help="Bearer token used to populate OTEL_EXPORTER_OTLP_HEADERS automatically.",
    )
    parser.add_argument(
        "--pr",
        help="Override the PR number (sets TRANSFORMERS_TEST_OTEL_PR and vcs.change.id).",
    )
    parser.add_argument(
        "--otlp-endpoint",
        "--oltp-endpoint",
        dest="otlp_endpoint",
        help="Override the OTLP endpoint URL without setting OTEL_EXPORTER_OTLP_ENDPOINT manually.",
    )
    parser.add_argument(
        "--staging-endpoint",
        dest="staging_endpoint",
        help=(
            "Additionally mirror every span to this second OTLP endpoint (e.g. a "
            "staging backend) on top of the primary --otlp-endpoint. Uses the same "
            "protocol as the primary."
        ),
    )
    parser.add_argument(
        "--staging-protocol",
        dest="staging_protocol",
        # No `choices` (unlike --protocol): CI passes this from a possibly-empty
        # env var, and an empty string must be tolerated as "inherit primary"
        # rather than rejected. Accepts "http"/"grpc"; empty/unset is a no-op.
        help=(
            "OTLP transport (http/grpc) for the --staging-endpoint. Falls back to "
            "the primary --protocol if omitted, so staging can use a different "
            "transport."
        ),
    )
    parser.add_argument(
        "--staging-token",
        dest="staging_token",
        help=(
            "Bearer token for the --staging-endpoint. Falls back to the primary "
            "token if omitted. Set independently so staging can use its own auth."
        ),
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
        job=args.job,
        service_name=args.service_name,
        force_export_traces=args.force_export_traces,
        protocol=args.protocol,
        otlp_endpoint=args.otlp_endpoint,
        staging_endpoint=args.staging_endpoint,
        staging_protocol=args.staging_protocol,
        token=args.token,
        staging_token=args.staging_token,
        pr=args.pr,
    )
    command = augment_pytest_command(command, export_traces=export_traces)
    env, trace_id = configure_trace_context(env, command, export_traces=export_traces)

    if args.print_config:
        print(
            json.dumps(
                {
                    "export_traces": export_traces,
                    "job": env.get("TRANSFORMERS_TEST_OTEL_JOB"),
                    "service_name": env.get("OTEL_SERVICE_NAME"),
                    "protocol": env.get("OTEL_EXPORTER_OTLP_PROTOCOL"),
                    "staging_endpoint": env.get(STAGING_ENDPOINT_ENV),
                    "staging_protocol": env.get(STAGING_PROTOCOL_ENV),
                    "traces_exporter": env.get("OTEL_TRACES_EXPORTER"),
                    "resource_attributes": env.get("OTEL_RESOURCE_ATTRIBUTES"),
                    "trace_id": trace_id,
                    "command": command,
                },
                sort_keys=True,
            )
        )

    if args.ping_server:
        ping_server(env)
        staging_endpoint = env.get(STAGING_ENDPOINT_ENV)
        if staging_endpoint:
            ping_server(
                env, endpoint=staging_endpoint, endpoint_source=STAGING_ENDPOINT_ENV
            )

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

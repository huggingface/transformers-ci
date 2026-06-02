import contextlib
import io
import json
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

from transformersci.otel import cli


class ConfigureCiOtelTests(TestCase):
    def test_prepare_environment_skips_export_without_endpoint(self):
        env, export_traces = cli.prepare_environment({})

        self.assertFalse(export_traces)
        self.assertNotIn("OTEL_SERVICE_NAME", env)
        self.assertNotIn("OTEL_RESOURCE_ATTRIBUTES", env)

    def test_prepare_environment_adds_github_attributes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as event_file:
            json.dump(
                {
                    "pull_request": {
                        "html_url": "https://github.com/huggingface/transformers/pull/4321",
                        "number": 4321,
                    },
                    "repository": {"full_name": "huggingface/transformers"},
                },
                event_file,
            )
            event_file.flush()

            env, export_traces = cli.prepare_environment(
                {
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_EVENT_PATH": event_file.name,
                    "GITHUB_JOB": "run_models_gpu",
                    "GITHUB_REF_NAME": "otel-support",
                    "GITHUB_RUN_ATTEMPT": "2",
                    "GITHUB_RUN_ID": "12345",
                    "GITHUB_SHA": "deadbeef",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                },
                job="models_gpu_slice",
            )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_SERVICE_NAME"], "transformers-tests")
        self.assertIn(
            "transformers.test.provider=github_actions", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn(
            "transformers.test.job=models_gpu_slice", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn("cicd.pipeline.run.id=12345:2", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn(
            "transformers.test.run.id=12345:2", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn(
            "transformers.test.job.run=12345:2:models_gpu_slice",
            env["OTEL_RESOURCE_ATTRIBUTES"],
        )
        self.assertIn("vcs.change.id=4321", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn(
            "vcs.change.url=https://github.com/huggingface/transformers/pull/4321",
            env["OTEL_RESOURCE_ATTRIBUTES"],
        )
        self.assertIn(
            "vcs.repository.name=huggingface/transformers",
            env["OTEL_RESOURCE_ATTRIBUTES"],
        )
        self.assertEqual(env["TRANSFORMERS_TEST_OTEL_RUN_ID"], "12345:2")

    def test_prepare_environment_adds_circleci_attributes(self):
        env, export_traces = cli.prepare_environment(
            {
                "CIRCLE_BRANCH": "pull/987",
                "CIRCLE_BUILD_NUM": "24680",
                "CIRCLE_JOB": "tests_torch",
                "CIRCLE_PROJECT_REPONAME": "transformers",
                "CIRCLE_PROJECT_USERNAME": "huggingface",
                "CIRCLE_PULL_REQUEST": "https://github.com/huggingface/transformers/pull/987",
                "CIRCLE_SHA1": "cafebabe",
                "CIRCLE_WORKFLOW_ID": "workflow-123",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
            },
            job="tests_torch",
        )

        self.assertTrue(export_traces)
        self.assertIn(
            "transformers.test.provider=circleci", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn(
            "transformers.test.job=tests_torch", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn(
            "cicd.pipeline.run.id=workflow-123", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn(
            "transformers.test.run.id=workflow-123", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn(
            "transformers.test.job.run=24680", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertIn("vcs.change.id=987", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn(
            "vcs.change.url=https://github.com/huggingface/transformers/pull/987",
            env["OTEL_RESOURCE_ATTRIBUTES"],
        )
        self.assertIn(
            "vcs.repository.name=huggingface/transformers",
            env["OTEL_RESOURCE_ATTRIBUTES"],
        )
        self.assertEqual(env["TRANSFORMERS_TEST_OTEL_RUN_ID"], "workflow-123")

    def test_prepare_environment_supports_local_forced_export(self):
        env, export_traces = cli.prepare_environment(
            {},
            job="local_smoke",
            force_export_traces=True,
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_SERVICE_NAME"], "transformers-tests")
        self.assertIn("deployment.environment=local", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn(
            "transformers.test.job=local_smoke", env["OTEL_RESOURCE_ATTRIBUTES"]
        )
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_grpc")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "grpc")

    def test_prepare_environment_uses_http_protobuf_for_https_endpoint(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
            }
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_http")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "http/protobuf")

    def test_prepare_environment_uses_env_otlp_endpoint(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
            }
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://otel.example")

    def test_prepare_environment_accepts_cli_otlp_endpoint(self):
        env, export_traces = cli.prepare_environment(
            {},
            otlp_endpoint="https://otel.example",
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://otel.example")
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_http")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "http/protobuf")

    def test_prepare_environment_cli_otlp_endpoint_overrides_traces_endpoint(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://old.example:4317",
            },
            otlp_endpoint="https://otel.example",
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://otel.example")
        self.assertNotIn("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", env)
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_http")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "http/protobuf")

    def test_prepare_environment_sets_headers_from_token(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
            },
            token="secret-token",
        )

        self.assertTrue(export_traces)
        self.assertEqual(
            env["OTEL_EXPORTER_OTLP_HEADERS"],
            "Authorization=Bearer secret-token",
        )
        self.assertEqual(
            env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"],
            "Authorization=Bearer secret-token",
        )

    def test_prepare_environment_sets_headers_from_env_token(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
                "OTEL_EXPORTER_OTLP_TOKEN": "env-token",
            }
        )

        self.assertTrue(export_traces)
        self.assertEqual(
            env["OTEL_EXPORTER_OTLP_HEADERS"],
            "Authorization=Bearer env-token",
        )
        self.assertEqual(
            env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"],
            "Authorization=Bearer env-token",
        )

    def test_prepare_environment_cli_token_overrides_env_token(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
                "OTEL_EXPORTER_OTLP_TOKEN": "env-token",
            },
            token="cli-token",
        )

        self.assertTrue(export_traces)
        self.assertEqual(
            env["OTEL_EXPORTER_OTLP_HEADERS"],
            "Authorization=Bearer cli-token",
        )
        self.assertEqual(
            env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"],
            "Authorization=Bearer cli-token",
        )

    def test_prepare_environment_normalizes_https_protocol_alias(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "https",
            }
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_http")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "http/protobuf")

    def test_prepare_environment_preserves_explicit_grpc_protocol(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            }
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_grpc")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "grpc")

    def test_prepare_environment_preserves_explicit_traces_exporter(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
                "OTEL_TRACES_EXPORTER": "otlp_proto_http",
            }
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_http")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "http/protobuf")

    def test_prepare_environment_protocol_argument_overrides_env(self):
        env, export_traces = cli.prepare_environment(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
                "OTEL_TRACES_EXPORTER": "otlp_proto_grpc",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            },
            protocol="http",
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp_proto_http")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "http/protobuf")

    def test_ping_server_skips_when_endpoint_is_missing(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            result = cli.ping_server({})

        self.assertFalse(result)
        self.assertIn("OTEL PING SKIPPED", stdout.getvalue())

    def test_ping_server_reports_success(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__exit__.return_value = False
        stdout = io.StringIO()

        with patch(
            "transformersci.otel.cli.socket.create_connection", return_value=connection
        ) as create_connection:
            with contextlib.redirect_stdout(stdout):
                result = cli.ping_server(
                    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel.example"},
                    timeout_seconds=0.1,
                )

        self.assertTrue(result)
        create_connection.assert_called_once_with(("otel.example", 4318), timeout=0.1)
        self.assertIn("OTEL PING OK", stdout.getvalue())

    def test_ping_server_uses_https_port_for_https_endpoint(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__exit__.return_value = False
        stdout = io.StringIO()

        with patch(
            "transformersci.otel.cli.socket.create_connection", return_value=connection
        ) as create_connection:
            with contextlib.redirect_stdout(stdout):
                result = cli.ping_server(
                    {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example"},
                    timeout_seconds=0.1,
                )

        self.assertTrue(result)
        create_connection.assert_called_once_with(("otel.example", 443), timeout=0.1)
        self.assertIn("OTEL PING OK", stdout.getvalue())

    def test_ping_server_reports_failure(self):
        stdout = io.StringIO()

        with patch(
            "transformersci.otel.cli.socket.create_connection",
            side_effect=OSError("connection refused"),
        ):
            with contextlib.redirect_stdout(stdout):
                result = cli.ping_server(
                    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel.example:4317"},
                    timeout_seconds=0.1,
                )

        self.assertFalse(result)
        self.assertIn("OTEL PING FAILED", stdout.getvalue())

    def test_augment_pytest_command_adds_export_flag_once(self):
        command = ["python3", "-m", "pytest", "tests"]

        augmented_command = cli.augment_pytest_command(command, export_traces=True)
        augmented_again = cli.augment_pytest_command(
            augmented_command,
            export_traces=True,
        )

        self.assertEqual(augmented_command[-1], "--export-traces")
        self.assertEqual(augmented_again.count("--export-traces"), 1)

    def test_configure_trace_context_generates_traceparent_for_pytest(self):
        env, trace_id = cli.configure_trace_context(
            {},
            ["python3", "-m", "pytest", "tests"],
            export_traces=True,
        )

        self.assertIsNotNone(trace_id)
        self.assertIn("TRACEPARENT", env)
        self.assertEqual(trace_id, cli.trace_id_from_traceparent(env["TRACEPARENT"]))

    def test_configure_trace_context_preserves_existing_traceparent(self):
        traceparent = "00-1234567890abcdef1234567890abcdef-fedcba0987654321-01"

        env, trace_id = cli.configure_trace_context(
            {"TRACEPARENT": traceparent},
            ["python3", "-m", "pytest", "tests"],
            export_traces=True,
        )

        self.assertEqual(env["TRACEPARENT"], traceparent)
        self.assertEqual(trace_id, "1234567890abcdef1234567890abcdef")

    def test_parse_args_accepts_otlp_flags(self):
        args = cli.parse_args(
            [
                "--protocol",
                "http",
                "--token",
                "secret-token",
                "--otlp-endpoint",
                "https://otel.example",
                "--job",
                "my_job",
                "--",
                "pytest",
            ]
        )

        self.assertEqual(args.protocol, "http")
        self.assertEqual(args.token, "secret-token")
        self.assertEqual(args.otlp_endpoint, "https://otel.example")
        self.assertEqual(args.job, "my_job")

    def test_parse_args_accepts_suite_alias_for_back_compat(self):
        args = cli.parse_args(
            [
                "--suite",
                "legacy_suite",
                "--",
                "pytest",
            ]
        )

        # --suite is kept as an alias for --job so existing workflows still work.
        self.assertEqual(args.job, "legacy_suite")

    def test_default_job_uses_local_default(self):
        self.assertEqual(cli.default_job({}, "local"), cli.DEFAULT_LOCAL_JOB)

    def test_job_run_id_for_github_actions(self):
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_RUN_ID": "999",
            "GITHUB_RUN_ATTEMPT": "1",
        }
        self.assertEqual(
            cli.job_run_id(env, "github_actions", "my_job"), "999:1:my_job"
        )

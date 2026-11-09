"""One timed, dashboard-compatible test span per MIC target and variant.

transformers-mic runs the cases in its own process and reports each one when it
finishes, so a case's span is opened from its result record and backdated by the
record's ``seconds``.
"""

import time

from transformersci.otel import instrument

CASE_ATTRIBUTES = (
    "contract",
    "contract_revision",
    "target",
    "kind",
    "revision",
    "variant",
    "framework",
    "fleet_repository",
    "fleet_revision",
    "selected_by",
)
RESULT_ATTRIBUTES = (
    "status",
    "platform",
    "weights",
    "source_model",
    "fixture_repo",
    "baseline",
    "baseline_repo",
    "baseline_revision",
    "baseline_path",
    "known_failure",
)


def nodeid(record):
    name = record.get("contract_name") or record["contract"].split("/")[-1]
    return f"tests/contracts::{name}::{record['target']}[{record['variant']}]"


class Telemetry:
    def __enter__(self):
        self.context = instrument.run("fast-integration")
        self.run = self.context.__enter__()
        return self

    def __exit__(self, kind, error, tb):
        if error is not None:
            # Setup/selection/download failures otherwise have no test span.
            self.runner_error(1, str(error))
        return self.context.__exit__(kind, error, tb)

    def runner_error(self, code, output):
        """A span for a run that failed before reporting any case."""
        with self.run.step("tests/contracts::runner") as step:
            step.set_exit_code(
                code, output=output or f"exit {code}", exception_type="MICRunnerError"
            )

    def result(self, record):
        attrs = {
            f"mic.{key}": str(record[key])
            for key in CASE_ATTRIBUTES + RESULT_ATTRIBUTES
            if record.get(key) is not None
        }
        if record.get("skipped_outputs"):
            attrs["mic.skipped_outputs"] = record["skipped_outputs"]
        seconds = record.get("seconds") or 0
        start = time.time_ns() - int(seconds * 1e9)
        with self.run.step(nodeid(record), attributes=attrs, start_time=start) as step:
            failed = record["status"] in ("fail", "error")
            step.set_exit_code(
                int(failed),
                command=f"{record['contract']} / {record['target']} ({record['variant']})",
                output="\n".join(record.get("failures", [])),
                exception_type="MICAssertionError"
                if record["status"] == "fail"
                else "MICExecutionError",
            )

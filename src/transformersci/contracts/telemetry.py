"""One timed, dashboard-compatible test span per MIC target and variant."""

from contextlib import contextmanager

from transformersci.otel import instrument


class Telemetry:
    def __enter__(self):
        self.context = instrument.run("fast-integration")
        self.run = self.context.__enter__()
        self.step = None
        return self

    def __exit__(self, kind, error, tb):
        if error is not None:
            # Setup/selection/download failures otherwise have no test span.
            with self.run.step("tests/contracts::runner") as step:
                step.set_exit_code(
                    1, output=str(error), exception_type="MICRunnerError"
                )
        return self.context.__exit__(kind, error, tb)

    @contextmanager
    def case(self, name, record):
        nodeid = f"tests/contracts::{name}::{record['target']}[{record['variant']}]"
        attrs = {
            f"mic.{key}": str(record[key])
            for key in (
                "contract",
                "contract_revision",
                "target",
                "kind",
                "revision",
                "variant",
                "framework",
                "selected_by",
            )
            if record.get(key) is not None
        }
        with self.run.step(nodeid, attributes=attrs) as step:
            self.step = step
            try:
                yield
            except BaseException as error:
                step.set_exit_code(
                    1, output=str(error), exception_type="MICRunnerError"
                )
                raise
            finally:
                self.step = None

    def result(self, record):
        if self.step is None:
            return
        for key in (
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
        ):
            if key in record:
                self.step.set_attribute(f"mic.{key}", str(record[key]))
        if record.get("skipped_outputs"):
            self.step.set_attribute("mic.skipped_outputs", record["skipped_outputs"])
        failed = record["status"] in ("fail", "error")
        self.step.set_exit_code(
            int(failed),
            command=f"{record['contract']} / {record['target']} ({record['variant']})",
            output="\n".join(record.get("failures", [])),
            exception_type="MICAssertionError"
            if record["status"] == "fail"
            else "MICExecutionError",
        )

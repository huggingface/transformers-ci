"""Run the pinned model integration contracts with transformers-mic, one OTEL span per case.

    run-model-contracts --fixtures-only --fleet control --variant default
    run-model-contracts --fixtures-only --fleet prototypes --variant bfloat16 \\
        --changed-files changed.txt --diff-base HEAD^1 --output results.jsonl
    run-model-contracts --lock /path/to/custom-fleet.json --fixtures-only

This is `tmic check` (transformers-mic, pinned in the `contracts` extra) plus the CI
telemetry: selection, execution, baseline comparison, known failures and result
records are all transformers-mic's. This wrapper only

  - pins the fleet to an immutable dataset revision before the run (an explicit
    --fleet-revision, else tests/contracts.lock, else main resolved once), so every
    record and span names the exact fleet it checked;
  - adds that fleet to each result record (fleet, fleet_repository, fleet_revision)
    or, with --lock, the lock path, so report-model-contracts can print a
    reproduction command;
  - turns each record into one test span (tests/contracts::<contract>::<target>[<variant>])
    under the fast-integration run, using the configure-ci-otel environment, so the
    cases appear on the PR dashboard like any other test;
  - appends the records to --output.

Records go to stdout, the runner's selection and summary go to stderr, and the exit
code is tmic's. Recording and reviewing baselines is authoring, not CI: use
`tmic record` / `tmic check` in a MIC workspace for that.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

PIN = Path("tests/contracts.lock")
STDERR_TAIL = 4000


def parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--framework",
        choices=("transformers", "sentence_transformers"),
        default="transformers",
        help="integration to run (tmic check --integration)",
    )
    parser.add_argument(
        "--contract", action="append", help="run only this contract (repeatable)"
    )
    parser.add_argument(
        "--target",
        action="append",
        help="run only this target, fixture or checkpoint name (repeatable)",
    )
    parser.add_argument(
        "--fixtures-only", action="store_true", help="skip real checkpoints"
    )
    parser.add_argument("--variant", default="default")
    parser.add_argument(
        "--isolate", action="store_true", help="run each target in its own process"
    )
    fleet = parser.add_mutually_exclusive_group()
    fleet.add_argument(
        "--fleet",
        choices=("control", "prototypes"),
        help="Hub fleet (default: control)",
    )
    fleet.add_argument("--lock", type=Path, help="custom JSON fleet lock to run")
    parser.add_argument(
        "--fleet-revision",
        help="dataset revision for --fleet (default: tests/contracts.lock, otherwise main)",
    )
    parser.add_argument(
        "--output", type=Path, help="append result records to this JSON lines file"
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        metavar="FILE",
        help="run only the contracts these changed Transformers files can affect (one path per line)",
    )
    parser.add_argument(
        "--diff-base",
        help="with --changed-files: base revision, so registry-only model additions select just those models",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="with --changed-files: run every contract anyway, recording selected_by",
    )
    parser.add_argument(
        "--run-declared",
        action="store_true",
        help="also run variants the contracts declare unmeasured",
    )
    return parser


def resolve_fleet(args):
    """(tmic options, metadata added to each record) for the fleet this run checks."""
    if args.lock:
        lock = str(args.lock.resolve())
        return ["--lock", lock], {"lock": lock}
    from transformers_mic import fleet

    name = args.fleet or "control"
    revision = fleet.resolve(
        args.fleet_revision or (fleet.read_pin(PIN) if PIN.exists() else "main")
    )
    print(
        json.dumps(
            {
                "fleet": {
                    "fleet": name,
                    "fleet_repository": fleet.REPOSITORY,
                    "fleet_revision": revision,
                }
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    return ["--fleet", name, "--fleet-revision", revision], {
        "fleet": name,
        "fleet_repository": fleet.REPOSITORY,
        "fleet_revision": revision,
    }


def check_command(args, fleet_options):
    command = [
        sys.executable,
        "-m",
        "transformers_mic.cli",
        "check",
        *fleet_options,
        "--integration",
        args.framework,
        "--variant",
        args.variant,
    ]
    for flag in ("fixtures_only", "isolate", "run_all", "run_declared"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    for name in args.contract or ():
        command += ["--contract", name]
    for name in args.target or ():
        command += ["--target", name]
    if args.changed_files:
        command += ["--changed-files", str(args.changed_files.resolve())]
        if args.diff_base:
            command += ["--diff-base", args.diff_base]
    return command


def result(line):
    """The result record on this stdout line, or None for anything else the run prints."""
    if not line.startswith("{"):
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(record, dict) and {"contract", "target", "status"} <= set(record):
        return record
    return None


def _main(argv, telemetry):
    args = parser().parse_args(argv)
    if args.lock and args.fleet_revision:
        parser().error("--lock cannot be combined with --fleet-revision")
    fleet_options, metadata = resolve_fleet(args)
    tail = []

    def relay_stderr(stream):
        # The runner's selection, summary and tracebacks stay on stderr (run.log in
        # CI); keep the end of it for a span if the run dies without a record.
        for line in stream:
            sys.stderr.write(line)
            sys.stderr.flush()
            tail.append(line)
            while sum(map(len, tail)) > STDERR_TAIL and len(tail) > 1:
                tail.pop(0)

    process = subprocess.Popen(
        check_command(args, fleet_options),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    relay = threading.Thread(target=relay_stderr, args=(process.stderr,), daemon=True)
    relay.start()
    records = 0
    for line in process.stdout:
        record = result(line)
        if record is None:
            sys.stdout.write(line)
            sys.stdout.flush()
            continue
        record = {**record, **metadata}
        telemetry.result(record)
        print(json.dumps(record), flush=True)
        if args.output:
            with args.output.open("a") as f:
                f.write(json.dumps(record) + "\n")
        records += 1
    code = process.wait()
    relay.join()
    if code and not records:
        telemetry.runner_error(code, "".join(tail).strip())
    return code


def main(argv=None):
    from .telemetry import Telemetry

    with Telemetry() as telemetry:
        return _main(argv, telemetry)


if __name__ == "__main__":
    sys.exit(main())

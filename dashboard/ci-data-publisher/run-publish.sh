#!/bin/sh
# One publish cycle: build the day partitions into the staging dir and push them
# to the HF bucket. Run by cron hourly and once at container startup.
set -eu

# Pull in the compose-provided env (cron's environment is otherwise near-empty).
[ -f /etc/publisher.env ] && . /etc/publisher.env

export PYTHONPATH="${PYTHONPATH:-/app/src}"

echo "[ci-data-publisher] $(date -u +%FT%TZ) cycle start"
python -m transformersci.publish.main --sync
echo "[ci-data-publisher] $(date -u +%FT%TZ) cycle done"

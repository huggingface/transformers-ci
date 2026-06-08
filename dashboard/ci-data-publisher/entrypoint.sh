#!/bin/sh
# Container entrypoint: persist the compose-provided env for cron, run one
# publish cycle immediately (so the bucket refreshes on deploy, not up to an
# hour later), then hand off to cron in the foreground.
set -eu

# cron jobs start with a near-empty environment, so snapshot the container's
# env into a file that run-publish.sh sources. Rendered as `export K="V"` lines;
# our values (URIs, tokens, durations) contain no embedded double quotes.
printenv \
  | sed 's/^\([A-Za-z_][A-Za-z0-9_]*\)=\(.*\)$/export \1="\2"/' \
  > /etc/publisher.env

echo "[ci-data-publisher] startup: running initial publish cycle"
/usr/local/bin/run-publish.sh || \
  echo "[ci-data-publisher] initial cycle failed (will retry on the hourly schedule)"

echo "[ci-data-publisher] starting cron (hourly)"
exec cron -f

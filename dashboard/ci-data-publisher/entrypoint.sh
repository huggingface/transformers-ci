#!/bin/sh
# Container entrypoint: persist the compose-provided env for cron, then hand off
# to cron in the foreground.
#
# We deliberately DON'T run a publish cycle on startup. The startup cycle used to
# fire on every container recreate (i.e. on every deploy) and fetch the whole
# window of large traces from Tempo at the same moment the live exporter is
# querying it — the concurrent load spiked the 2 GiB single-node Tempo, timed out
# the exporter's renders, and briefly blanked the dashboard. Letting the hourly
# cron own the schedule means only one consumer hits Tempo hard at a time. The
# cost: after a deploy the bucket refreshes at the next :17 cron rather than
# immediately — run `docker compose exec ci-data-publisher run-publish.sh` if you
# need an on-demand publish.
set -eu

# cron jobs start with a near-empty environment, so snapshot the container's
# env into a file that run-publish.sh sources. Rendered as `export K="V"` lines;
# our values (URIs, tokens, durations) contain no embedded double quotes.
printenv \
  | sed 's/^\([A-Za-z_][A-Za-z0-9_]*\)=\(.*\)$/export \1="\2"/' \
  > /etc/publisher.env

echo "[ci-data-publisher] starting cron (publishes hourly at :17)"
exec cron -f

# Publisher sidecar: shapes Tempo traces into the public CI-telemetry bucket
# and pushes them hourly with `hf sync`. Unlike the trace-exporter (stdlib
# only, so it runs on a bare python:3.12-slim), this needs pyarrow + the hf
# CLI + a real cron, hence its own image. App code is bind-mounted from
# ../src at runtime (PYTHONPATH=/app/src), so logic changes need no rebuild.
FROM python:3.12-slim

# cron for the hourly schedule; ca-certificates for HTTPS to the Hub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cron ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Xet-backed sync + Parquet writing. hf_xet gives the bucket's dedup storage.
RUN pip install --no-cache-dir "huggingface_hub[cli,hf_xet]" pyarrow

COPY crontab /etc/cron.d/ci-data-publisher
COPY run-publish.sh /usr/local/bin/run-publish.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0644 /etc/cron.d/ci-data-publisher \
    && chmod 0755 /usr/local/bin/run-publish.sh /usr/local/bin/entrypoint.sh \
    && crontab /etc/cron.d/ci-data-publisher

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

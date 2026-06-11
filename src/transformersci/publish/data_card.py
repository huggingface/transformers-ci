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
"""Render the dataset's data card (``README.md``) published into the bucket.

Documents the schema, the ``model``/``gpu``/``status_code`` derivations, the
partition layout, and load examples, so the bucket is self-describing for
anyone building on top of it.
"""

from __future__ import annotations

from .tables import RUN_ROLLUP_COLUMNS, SCHEMA_VERSION, TEST_ROW_COLUMNS


def render_data_card(bucket_uri: str) -> str:
    """Return the Markdown data card for the published bucket."""
    test_cols = "\n".join(f"- `{c}`" for c in TEST_ROW_COLUMNS)
    rollup_cols = "\n".join(f"- `{c}`" for c in RUN_ROLLUP_COLUMNS)
    # The hf://buckets/<ns>/<name> URI maps to a glob path under the same name.
    path_root = bucket_uri.rstrip("/")
    return f"""# transformers CI telemetry

Public, daily-partitioned snapshot of the transformers CI test telemetry
collected by the pytest observability stack (OpenTelemetry → Tempo). Refreshed
**hourly**. Schema version **v{SCHEMA_VERSION}**.

This is derived from raw test-execution traces so you can build apps and
analyses on top of CI data without access to the internal stack.

## Layout

```
current_view.json          manifest: schema_version, updated_at, partitions, totals
README.md                  this data card
daily/
  <YYYY-MM-DD>/            partition = UTC day of the run's start
    test_rows.parquet       one row per (trace_id, test_nodeid)
    run_rollups.parquet     one row per (run_id, test_job)
    traces/
      <trace_id>.json       raw Jaeger-shaped trace (full fidelity)
```

The bucket is the long-term archive; it keeps full history independent of the
stack's operational retention.

## `test_rows` columns

{test_cols}

`model` is derived from `tests/models/<model>/...` nodeids (empty otherwise).
`gpu` is derived from the job name (`single` / `multi` / empty).
`status_code` is the OTEL span status: `OK` / `ERROR` / `UNSET`.
`exception_message` and `exception_stacktrace` are the **full, untruncated**
failure text.

## `run_rollups` columns

{rollup_cols}

`job_count` is the number of distinct jobs that contributed tests to the run.

## Load examples

```python
import pandas as pd
df = pd.read_parquet(
    "{path_root}/daily/2026-06-08/test_rows.parquet"
)
```

```sql
-- DuckDB, straight from the bucket
SELECT model, count(*) FILTER (WHERE status_code = 'ERROR') AS fails
FROM '{path_root}/daily/*/test_rows.parquet'
GROUP BY model ORDER BY fails DESC;
```

## Notes

- Coverage reflects what the stack actually traced — uninstrumented workflows
  are absent here too.
- Columns are additive across schema versions; `current_view.json.schema_version`
  is bumped on any change.
"""

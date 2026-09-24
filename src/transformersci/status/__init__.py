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
"""Live CI run/job status from GitHub, independent of test telemetry.

The trace pipeline only learns about a job once it emits a span, which is after
queueing, container start and checkout, and it publishes on a minutes-long
render cycle. This package follows GitHub's own ``workflow_run`` and
``workflow_job`` events instead:

* :mod:`.webhook` verifies a delivery and turns it into updates;
* :mod:`.reducer` merges an update into the current state (pure, ordered);
* :mod:`.store` keeps that state durably in SQLite;
* :mod:`.metrics` renders it as ``ci_github_*`` Prometheus series;
* :mod:`.server` / :mod:`.cli` serve the webhook and ``/metrics``.

Stdlib only, Python 3.10.
"""

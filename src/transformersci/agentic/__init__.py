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
"""Agentic CI tooling — daily failure triage and auto-fix dispatch.

Holds the consumer-side pipeline that reads the daily CI dataset, clusters the
persistent integration-test failures, and dispatches one fix task per failure
group to Serge. Exposed as the ``integration-failure-triage`` console script
and driven from a reusable GitHub Actions workflow.
"""

# MIC production validation — 2026-10-20

Post-fix merge-queue runs [39414851656](https://github.com/huggingface/transformers/actions/runs/39414851656)
and [39416548925](https://github.com/huggingface/transformers/actions/runs/39416548925)
used runner `b27b945` and produced identical case results: 18 pass and four fail.
The control fixture and reviewed Hub references loaded successfully. The job
remained green while recording assertion errors, as intended.

| Case | Observation on both runs |
| --- | --- |
| Whisper | Reviewed baseline passes; the old expected-failure entry is stale. [#49456](https://github.com/huggingface/transformers/issues/49456) is closed. |
| Wav2Vec2 | `logits`: maximum delta 0.000977 exceeds the reviewed tolerance. |
| Mistral3 | `answer_logits`: maximum delta 0.00488 exceeds the reviewed tolerance. |
| SmolVLM | `selected_logits`: maximum delta 0.00195 exceeds the reviewed tolerance. |

The runtime differs from the baseline environment: Python 3.11.17 versus 3.10,
Torch 2.14.0 CPU versus 2.11.0, and NumPy 2.3.5 versus 1.26.4, among other
packages. The repeated numeric deltas establish reproducibility on these CI
runs; they do not establish whether the cause is framework code or runtime
drift. Keep the assertions and reviewed tolerances unchanged pending a replay
under the recorded baseline environment. The earlier Wav2Vec2 non-finite
training result did not recur in these rows; the lock still declares the
existing training-output exemption separately from its logits assertion.

The runner stopped before the final two prototypes because its integration
schema rejected GPT-OSS's reviewed `config_overrides` build field. The schema
update in [#327](https://github.com/huggingface/transformers-ci/pull/327) accepts
that field and the current fixture provenance options without allowing arbitrary
build fields.

Production Tempo contains both control and prototype traces, and Prometheus
indexes their case and runner failures. The prototype trace
`f068b04d95d96abc9b257c2f06f289f9` has 23 spans but only four unique span IDs:
usage reseeds the global RNG that the default OTEL ID generator uses. #327
applies the existing seed-independent generator. Its regression test verifies
unique IDs across repeated model seeding.

The MIC page was replayed against that trace and its production failure index.
It produces four case rows with assertion text and pinned baseline links, plus
the runner schema error. Its persistence path stores MIC metadata separately
from large pytest run files. Full fleet completion and unique IDs in a fresh
production run still need verification after #327 lands.

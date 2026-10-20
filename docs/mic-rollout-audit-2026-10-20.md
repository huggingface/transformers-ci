# MIC rollout audit, 2026-10-20

Downloaded and inspected all 531 available `contracts` artifacts, from October 9
through October 20 at 08:40 UTC, with zero download errors. Of these, 465 were
saved after the production rollout merged on October 15 at 15:39:54 UTC. They
contain 8,675 case records. This covers saved artifacts, not jobs that stopped
before uploading artifacts. Counts below are artifact occurrences, not distinct
bugs. The accompanying JSON contains aggregate evidence and example run links.

Since rollout, the saved observations report 375 failed run steps, 89 successful
steps and one skipped step. There are 93 empty result artifacts; emptiness alone
is not a failure, since an unrelated change may select no contracts. The GitHub
job conclusion cannot establish whether its cases passed: it is observational.

| Finding | Occurrences since rollout | Evidence |
| --- | ---: | --- |
| Whisper passes but a stale known-failure entry turns that into a failure | 360 | [Latest main run](https://github.com/huggingface/transformers/actions/runs/39392740233) |
| SmolVLM invalid view shape `[-1, 14, 64]` for 176256 elements | 182 | [Release branch example](https://github.com/huggingface/transformers/actions/runs/39295399702) |
| SmolVLM `not enough values to unpack (expected 3, got 2)` | 1 | [PR run](https://github.com/huggingface/transformers/actions/runs/38799169278) |
| Kernels dependency check fails all 24 cases | 1 run / 24 cases | [Run](https://github.com/huggingface/transformers/actions/runs/39263753278) |

SmolVLM's last saved main-branch shape failure was October 19 at 12:11 UTC
([run](https://github.com/huggingface/transformers/actions/runs/39249737079)). It passes
in the latest saved main run at October 20 08:24 UTC. Failures on older branches
continued later. This establishes recovery in the observed execution path, not
numeric equivalence: the deployed command used `--no-baseline`.

Three failed steps had no result because `tests/contracts/prototype-fleet.lock`
was missing from the framework checkout ([example](https://github.com/huggingface/transformers/actions/runs/38762534850)).
Other failure logs include an empty-selection/explicit-target exit failure,
a broken output pipe ([run](https://github.com/huggingface/transformers/actions/runs/38795464104)),
and a cancelled run ([run](https://github.com/huggingface/transformers/actions/runs/39257835836)).
One failed observation had no runner log; its artifact alone cannot establish a cause.

The current image also differs substantially from the historical baseline runtime:
the latest main artifact reports Python 3.11.17, Torch 2.14.0+cpu and NumPy 2.3.5,
versus lock pins Python 3.10, Torch 2.11.0 and NumPy 1.26.4. New numeric failures
must be examined with this provenance; they cannot automatically be attributed to
a Transformers change. No baseline values or tolerances were changed during this audit.

The replacement runner bundles its locks, fixes empty selections, accepts OTEL
wrapper output in the log pipeline, and restores numeric and exact assertions.
It deliberately retains strict known-failure handling: the old Whisper entry
will ask for review when the baseline check passes rather than silently hiding
an unexpected pass.

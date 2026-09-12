# Research benchmark: protocol, results, and provenance

[Documentation index](README.md) · [Execution runbook](../experiments/RESEARCH_BENCHMARK.md) · [Artifact guide](../results/README.md)

The research benchmark compares the frozen MNEMA implementation with five dense continual-learning baselines on Split-MNIST. This page defines the comparison and records the separately supplied full-data results. Use the runbook for commands and the saved configuration for a particular run's exact settings.

## Evidence status

The repository tracks a **reduced, one-seed debug bundle** under [results/research_benchmark/quick](../results/research_benchmark/quick/). A completed full-data bundle was supplied separately in the parent `Desktop/cognx_benchmarking` directory. Its status, validation record, and summary were inspected during this documentation revision; the full experiment was not rerun.

A fresh clone does not include that external bundle. The [full-data table below](#recorded-full-data-results) is a transcription with explicit provenance, not a substitute for the raw evidence. Historical results under `results/` use a different protocol and should not be combined with either research result family.

## Protocol

| Dimension | Full research setting |
| --- | --- |
| Dataset | All 60,000 MNIST training images and all 10,000 test images |
| Task sequence | Five digit pairs: `01 -> 23 -> 45 -> 67 -> 89` |
| Inference | All ten output classes; no task identifier or output masking |
| External training | One presentation of each training example, with a task stream shuffled once per seed |
| Replication | Seeds 0-9, six methods per seed, 60 independent seed-method pairs |
| Evaluation | All five test tasks at each task checkpoint; all 25 retention cells measured |
| Compute environment | NumPy on CPU; runner sets BLAS-related thread counts to one |
| Reference software | Python 3.11.7, NumPy 2.2.6, Matplotlib 3.10.3, PyYAML 6.0.2 |
| Selection | Fixed defaults, without final-test-set hyperparameter tuning |

Each method receives the same per-seed training order and test indices. Images remain uint8 in shared host arrays and are normalized to float32 per use. Compressed MNIST files are checked against expected MD5 values; their SHA-256 digests are saved in configuration. See [research_split_mnist.py](../benchmarks/research_split_mnist.py).

One external pass does not mean equal computation or strictly one data access: EWC makes additional training-data reads for Fisher estimation; replay methods and MNEMA consolidation perform internal replay.

Quick mode defaults to one seed and 100 train / 100 test images per task. It retains the same model dimensions and explicitly labels results as debug-only.

## Methods

All dense baselines share a `784 -> 256 ReLU -> 10` float32 MLP, the same initial parameters for a given seed, and SGD with learning rate 0.01. There is no optimizer momentum or pretrained backbone.

| Slug | Report name | Update and auxiliary state |
| --- | --- | --- |
| `naive` | Naive MLP | Cross-entropy update on the incoming sample; no replay |
| `replay300` | Replay-300 (Research) | True 300-item reservoir of uint8 images/labels; one update on mean CE of current plus one prior sample; insertion follows sampling |
| `replay64kib` | Replay-64KiB | Same policy, with capacity derived from a strict 65,536-byte buffer payload budget including counters; 83 items at these dimensions |
| `ewc` | EWC | CE plus task-wise diagonal Fisher penalty; lambda 100; parameter snapshots and Fishers from the first four tasks |
| `derpp300` | DER++-300 | 300-item reservoir containing images, labels, and ten float32 logits captured before the original optimizer update |
| `mnema` | MNEMA | Frozen model defaults, adaptive sparse representation, FastStore, cortex, and internal sleep consolidation |

The main runner accepts `replay64k` as an alias for `replay64kib`, and `derpp` for `derpp300`. Distributed helper commands use canonical slugs.

EWC uses the mean of squared per-example CE gradients with observed training labels as an empirical diagonal Fisher. The penalty is `lambda / 2` times the sum of Fisher-weighted squared parameter deviations over previous tasks. By default every training sample contributes to each of the first four task Fishers. This pass makes no optimizer updates, and no Fisher is built after the final task.

DER++ uses `CE(current) + 0.5 * mean(logit_error^2) + 0.5 * CE(replayed_label)`. The logit-matching and label-replay terms draw independent prior examples and contribute to one combined update. These coefficients are fixed illustrative defaults, not a claim of optimal tuning.

Historical replay in [baselines/mlp.py](../baselines/mlp.py) instead uses random replacement and different insertion/update ordering. Its numbers do not describe the research reservoir baseline.

## Evaluation isolation

MNEMA's native prediction path changes adaptive state. The research harness wraps each trained checkpoint in [CheckpointInference](../experiments/research_state.py), which restores encoder, separator, cortex, controller, and RNG state after every test image, including when inference raises an exception.

The checkpoint's existing membrane state is preserved rather than zeroed. Full attribute-state hashes are checked before and after each evaluation set. At the final checkpoint, reverse-order test predictions must match image by image. Predictions receive no labels or task identifiers; labels are used afterward for scoring.

This is an evaluation-layer protocol. It does not modify MNEMA's source or imply that the webcam demonstration has stateless inference.

## Metrics

Let `R[t, j]` be accuracy as a fraction on task `j` after training task `t`, with indices 0 through 4.

```text
Final ACC (%) = 100 * mean(R[4, j] for j = 0,...,4)

Forgetting (pp) = 100 * mean(
    max(R[t, j] for t = j,...,3) - R[4, j]
    for j = 0,...,3
)
```

Final accuracy is a task-macro average: task test sets differ slightly in size, so it is not pooled sample accuracy. Forgetting excludes pre-learning checkpoints, the final checkpoint from the maximum, and the fifth task. Negative forgetting is allowed when final performance improves beyond the earlier best score.

The unit of statistical replication is the seed. Tables report the mean and sample SD (`ddof=1`), not standard error or a confidence interval. A single seed has undefined SD. Low forgetting should always be read alongside accuracy: a model that learned little may have little to forget. No significance claim follows from these descriptive summaries.

## Memory and energy accounting

### Memory

Measurements count unique persistent NumPy array payloads:

```text
Total resident arrays = model/adaptive + fixed scaffold + auxiliary allocated
```

Active auxiliary content is an alternative occupancy view and is **not added again**. Buffer size/seen counters are explicit int64 arrays and are counted. Excluded items include Python object/scalar overhead, PRNG internals, shared data, temporary gradients/Fisher workspace, and evaluation snapshots. These values are not process RSS or peak deployment memory.

For MNEMA, cortex synaptic planes, eligibility traces, neuron state, and separator homeostasis belong to adaptive state. Separator connectivity and synaptic constants belong to fixed scaffold. FastStore's default active row is 28 bytes, while eviction counts 25; its configured 64 KiB budget is therefore not an actual active-payload ceiling. Full allocated FastStore capacity is 458,752 bytes. Active rows do not constitute a demonstrated packed deployment format.

Replay-64KiB's episodic buffer is strictly budgeted, but it is not an exact memory match to MNEMA. Dense model parameters and other state remain additional memory. All research replay buffers have fixed capacities.

### Energy and runtime

Native inference counters are projected using the [ASIC 45 nm card](../instrument/tech/asic_45nm.yaml). Projections omit some arithmetic and traffic terms, plus checkpoint restoration. In particular, the instrument records write counters but its projection formula does not charge writes. Technology cards describe cost assumptions, not executed target devices.

The research baselines' training, EWC Fisher estimation, and buffer work do not have comparable complete energy instrumentation. The report therefore makes **no cross-method end-to-end training-energy ranking**. Native MNEMA training counts remain in raw records. Wall time is measured on the host, including integrity work; it is not measured energy.

## Recorded full-data results

The separately supplied bundle records:

| Provenance field | Recorded value |
| --- | --- |
| Status | `complete`; `research_complete: true` |
| Coverage | 60 seed-method results; seeds 0-9, all six methods |
| Validation | All 60 saved-artifact checks valid |
| Configuration ID | `ab869e57feea62e93198f72fedde18182dc93a28a72d4875e966067d676027a8` |
| Source commit | `fe86ec1c6abc600dda8ec50565a551af4e5434bd` |
| Software | Python 3.11.7 / NumPy 2.2.6 / Matplotlib 3.10.3 / PyYAML 6.0.2 |

Values below are transcribed from its `summaries/summary_table.md`. Accuracy and forgetting are mean +/- sample SD over ten seeds. Memory and inference projections shown are the reported means.

| Method | Final ACC (%) | Forgetting (pp) | Resident arrays (bytes) | Projected native inference (microjoules/image) |
| --- | ---: | ---: | ---: | ---: |
| Naive MLP | 19.79 +/- 0.06 | 99.52 +/- 0.12 | 814,120 | 1.768 |
| Replay-300 (Research) | 82.44 +/- 1.33 | 20.58 +/- 1.62 | 1,049,636 | 1.768 |
| Replay-64KiB | 67.59 +/- 2.07 | 39.30 +/- 2.60 | 879,291 | 1.768 |
| EWC | 19.98 +/- 0.36 | 99.12 +/- 0.40 | 7,327,080 | 1.768 |
| DER++-300 | 89.39 +/- 0.60 | 12.12 +/- 0.73 | 1,061,636 | 1.768 |
| MNEMA | 77.12 +/- 0.87 | 6.53 +/- 1.41 | 4,391,100 | 0.316 |

MNEMA has lower observed forgetting than the other methods in this record, while Replay-300 and DER++-300 have higher final accuracy. MNEMA also allocates substantially more resident array payload than either replay method. The smaller native inference projection must be interpreted within the incomplete counter boundary above.

Retain the external bundle's `config.json`, `status.json`, `validation.json`, `full_results.json`, `raw/`, `workers/`, `summaries/`, and `plots/` together. Without it, a reader can inspect the protocol and debug artifacts but cannot independently audit this full-data table from the clone alone.

## GitHub Actions

The included workflows are manual-only (`workflow_dispatch`); ordinary pushes do not launch benchmark jobs. Their configuration is described here from the checked-in YAML, not from a newly executed cloud run.

| Workflow | Selection | Intended work |
| --- | --- | --- |
| [Research Benchmark](../.github/workflows/research-benchmark.yml) | `smoke/full-seed-0` | Six full-data method jobs for seed 0; no final matrix aggregation |
| Same workflow | `full-10-seed` | 60 jobs followed by full aggregation |
| [Research Benchmark Pair](../.github/workflows/research-benchmark-pair.yml) | Seed 0-9 and one canonical method | One full-data pair |

The matrix sets `fail-fast: false`, `max-parallel: 6`, and a 350-minute benchmark-job timeout. Preparation installs the research pins, verifies MNIST, and uploads shared inputs. A failed pair does not cancel other pair jobs; successful aggregation still requires the expected inputs.

**Known workflow path mismatch:** both YAML files translate the `derpp300` directory to `derpp`, but the current Python pair runner writes to `results/github-actions/seed_<seed>/derpp300/`. The matrix then tries to copy from the wrong directory, preventing successful DER++ artifact assembly and full aggregation as written. The pair workflow's log destination is also affected. This documentation revision does not change workflow execution. Align the YAML paths with the runner before relying on the full cloud matrix; the local canonical pair command in the runbook writes to the correct path.

When operating an aligned workflow, open the repository's **Actions** tab, choose the workflow, and select **Run workflow** on the intended source revision. Pair artifact names follow `research-seed-<seed>-<method>`; the YAML currently names DER++ artifacts with `derpp`. The full aggregation artifact is `research-benchmark-final`. Configured retention is 30 days, so download needed bundles before expiry.

The distributed [research_actions.py](../experiments/research_actions.py) helper also provides `prepare`, `run`, and `collect` commands. Its collector requires one unique raw result and matching worker record per expected pair, and verifies source, software, dataset, configuration, and indices. Pair output from the main runner alone is not a complete distributed shard. See the [runbook](../experiments/RESEARCH_BENCHMARK.md#distributed-helper) for a compatible example.

## Limits of the comparison

The experiment covers one dataset, one task order, simple dense baselines, fixed hyperparameters, and a shallow prototype. EWC has training task boundaries and additional Fisher reads; methods have unequal internal computation. Energy is projected, memory is array payload, and exact cross-platform floating-point equivalence is not guaranteed.

Historical evaluation let test inputs alter MNEMA state, left unmeasured future retention cells as zero, and used a different forgetting maximum. Dataset size, order, replay policy, and evaluation all changed between historical and research protocols. Differences in their scores cannot be attributed solely to an improvement in MNEMA.

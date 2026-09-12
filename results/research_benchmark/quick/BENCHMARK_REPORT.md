# MNEMA Continual-Learning Benchmark

[Artifact guide](../../README.md) · [Research protocol](../../../docs/RESEARCH_BENCHMARK.md) · [Reproduction commands](../../../experiments/RESEARCH_BENCHMARK.md)

This is the saved **quick-run report**, generated from this directory's JSON. Editorial navigation and context have been added; regenerating the report can replace those additions. Numerical results are unchanged. The separately supplied full-data record is documented in the research protocol and is not contained in this bundle.

**DEBUG RESULTS ONLY — reduced data, not research evidence. The full ten-seed experiment has NOT been completed by this quick run.**

† One seed: standard deviation is undefined; values are single-seed observations.

## 1. Research Question

How does the existing MNEMA compare with standard continual-learning strategies in accuracy, forgetting and memory usage under sequential Split-MNIST Class-IL?

## 2. Experimental Setup

MNIST handwritten digits, fixed task order 01 → 23 → 45 → 67 → 89. Each model receives the same shuffled training indices once per seed. Every prediction considers all ten classes; no inference task ID, output masking or task-specific head is used. All five test tasks are evaluated after every training task, including future-task cells; progress averages only tasks seen so far. The full mode uses 60,000 training and 10,000 test images. This run uses 500 training and 500 test images.

| Digits | Used train | Used test | Full train | Full test |
|---|---:|---:|---:|---:|
| 01 | 100 | 100 | 12,665 | 2,115 |
| 23 | 100 | 100 | 12,089 | 2,042 |
| 45 | 100 | 100 | 11,263 | 1,874 |
| 67 | 100 | 100 | 12,183 | 1,986 |
| 89 | 100 | 100 | 11,800 | 1,983 |

Seeds: [0]. Run mode: **quick**. Created: 2026-09-11T15:48:27.019381+00:00. Git commit: `bb5fd7d6318fa1109d462317436087718d6d1ae9`. Exact source hashes, dataset checksums and settings: [config.json](config.json). Software: `{'python': '3.11.7', 'numpy': '2.2.6', 'matplotlib': '3.10.3', 'pyyaml': '6.0.2', 'platform': 'Windows-10-10.0.22621-SP0', 'processor': 'Intel64 Family 6 Model 126 Stepping 5, GenuineIntel', 'blas_threads': 1}`. NumPy/CPU, one BLAS thread, sequential methods/seeds; no GPU or neuromorphic device assumed. Images remain uint8 in host memory and are normalized to float32 /255 per use.

The unit of statistical replication is the seed. Tables use sample standard deviation (ddof=1); SD is undefined for one seed, so quick figures omit error bars. Source/environment/configuration mismatches refuse resume; valid completed seed–method pairs are skipped. No statistical significance or tuned-best-method claim is made.

## 3. Methods

### Naive MLP
Normal online SGD; 784 → 256 ReLU → 10, fp32, learning rate 0.01. Same architecture and initialization per seed for all dense baselines. No optimizer momentum or pretrained weights.

### Replay-300 (Research)
Stores 300 uint8 images and labels using true reservoir sampling. Each incoming sample produces one update on mean CE over current plus one previous sample, then enters the reservoir. Historical replay instead used random replacement, insertion before sampling and separate SGD steps; it is preserved but is not included under this research method's name.

### Replay-64KiB
Same research replay policy with a strict 65,536-byte allocated array-payload buffer budget, including image, label and seen/size counters. Capacity is derived from NumPy dtype sizes. Inputs are stored without float inflation. This is a conventional replay method under a 64 KiB episodic-buffer constraint, **not exact memory matching to MNEMA**.

### EWC
Protects parameters important to past tasks using lambda/2 times the sum of diagonal Fisher-weighted squared deviations from each task's optimum. Lambda=100.0. Stores separate old-task Fishers and parameter snapshots. The empirical Fisher averages squared individual cross-entropy gradients using observed training labels. Fisher sample limit=0 (0=all training samples); actual counts per run: [[100, 100, 100, 100]]. This additional training-data pass makes no optimizer updates; it is reported compute overhead and is not a strictly single-access stream algorithm. No Fisher is built after the last task because no future training uses it. [Original EWC paper](https://arxiv.org/abs/1612.00796).

### DER++-300
Reservoir of uint8 images, labels and ten fp32 logits produced before the original sample's optimizer update. Loss is current CE + 0.5 × mean squared logit error + 0.5 × replay-label CE. The two replay terms draw independent single examples and share one combined update. Defaults are fixed illustrative coefficients, not test-tuned optimum values. [DER++ paper](https://arxiv.org/abs/2004.07211), [authors' reference implementation](https://github.com/aimagelab/mammoth/blob/master/models/derpp.py).

### MNEMA
The unmodified implementation combines a temporal spike encoder, sparse random separator with homeostasis, FastStore associations, a shallow adaptive spiking cortex with Benna–Fusi variables, arbitration, neuromodulation and internal sleep consolidation. Dimensions and thresholds remain unchanged. Internal consolidation adds replay computation; one pass refers to external stream presentation, not equal compute across algorithms.

## Implementation Caveats Discovered During Audit

### Stateful evaluation issue
The original historical benchmark evaluated the live model; held-out inputs could change adaptive state and subsequent training. **MNEMA's native inference path is stateful. The research benchmark therefore restores the same trained checkpoint state after every test inference so held-out evaluation data cannot alter subsequent predictions or training.** This is an evaluation protocol, not a modification to MNEMA. Restored attributes: encoder threshold/reference; separator thresholds/running rates; cortex membrane/adaptation; controller bus; RNG states. A full attribute-state hash is checked before/after every evaluation set. All final-checkpoint test sets are additionally predicted in reverse order and compared image by image. The checkpoint's existing membrane state is preserved, not reset to zero. No per-image full-model deepcopy is used.

### FastStore accounting issue
The frozen implementation counts 25 bytes/occupied row, whereas its int16 weights, float32 salience and uint32 timestamp require 28 bytes/row. Consequently the configured 64 KiB limit is **not a true physical 64 KiB payload ceiling**. The benchmark measures actual values and flags exceedances without changing eviction. The 64 KiB configuration also does not include cortex or separator state. Historical results are not overwritten or directly comparable to this protocol.

## 4. Main Results

| Method | Final ACC (%) | Forgetting (pp) | Model/adaptive (B) | Fixed scaffold (B) | Auxiliary content (B) | Auxiliary allocated (B) | Total resident arrays (B) | Native projected inference (µJ/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive MLP | 19.40† | 98.25† | 814120† | 0† | 0† | 0† | 814120† | 1.768† |
| Replay-300 (Research) | 65.40† | 37.50† | 814120† | 0† | 235516† | 235516† | 1049636† | 1.768† |
| Replay-64KiB | 67.00† | 35.00† | 814120† | 0† | 65171† | 65171† | 879291† | 1.768† |
| EWC | 19.60† | 98.00† | 814120† | 0† | 6512960† | 6512960† | 7327080† | 1.768† |
| DER++-300 | 72.00† | 26.75† | 814120† | 0† | 247516† | 247516† | 1061636† | 1.768† |
| MNEMA | 71.20† | 24.25† | 3408032† | 524316† | 73388† | 458752† | 4391100† | 0.316† |

All memory values are bytes (KiB = 1,024 bytes). Auxiliary content and allocated auxiliary memory are alternative views, not additive columns. Total = adaptive + fixed + allocated auxiliary. Energy is a partial native **projection**, not measured physical energy; see Section 8.

## 5. Accuracy

Final ACC is the unweighted average of the five final task accuracies. DER++-300 has the highest observed mean (72.00%). Task sizes differ slightly, so this is a task-macro average rather than pooled sample accuracy. This ranking is only a pipeline smoke-test observation.

![Final accuracy](plots/final_accuracy.png)

## 6. Forgetting

For zero-based task j=0,...,3, F_j = max(R[t,j] for t=j,...,3) − R[4,j]. Average these four values and multiply by 100 to report percentage points. The final checkpoint and checkpoints before learning j are excluded from the maximum. Negative forgetting is allowed and means final performance improved beyond previous post-learning checkpoints. The fifth task has no later learning period and is excluded. MNEMA has the lowest observed mean forgetting (24.25 pp). Low forgetting with low accuracy can mean a model never learned much; read both metrics together.

![Retention matrices](plots/retention_matrices.png)

## 7. Memory Trade-off

Measurements are unique persistent NumPy array payload, not interpreter/process RSS. Python object headers, scalar objects, PRNG internals, shared dataset, temporary gradient/Fisher workspace and evaluation snapshots are excluded for all methods. Buffer seen/size metadata are explicitly stored and counted as int64 arrays. MNEMA scalar adaptive state is restored and hashed even though scalar-object memory is outside this array-payload accounting. Fixed scaffold includes separator connectivity and synapse constants. Cortex synaptic variables, eligibility traces, neuron arrays and separator adaptive arrays are model/adaptive state, not frozen memory.

FastStore content counts rows with salience > 0, including weights, salience and timestamps. Unused dense capacity and stale timestamps are included in allocated FastStore arrays. Active content is not a claim of a deployable packed sparse representation; such a representation could need extra row-address metadata.

| MNEMA quantity | Final mean ± SD across seeds | Maximum across seeds |
|---|---:|---:|
| configured_budget_bytes | 65536† | 65,536 |
| implementation_row_bytes | 25† | 25 |
| actual_row_bytes | 28† | 28 |
| active_rows | 2621† | 2,621 |
| implementation_occupancy_bytes | 65525† | 65,525 |
| actual_active_payload_bytes | 73388† | 73,388 |
| allocated_faststore_bytes | 458752† | 458,752 |
| cortex_synaptic_adaptive_bytes | 3276960† | 3,276,960 |
| other_adaptive_array_bytes | 131072† | 131,072 |
| fixed_scaffold_bytes | 524316† | 524,316 |
| total_resident_array_bytes | 4391100† | 4,391,100 |
| peak post-sample active payload | 73388† | 73,388 |

Post-sample over-budget observations across runs: **370**. Monitoring occurs after each external training sample (including its consolidation); the maximum is not a bound on transient within-step occupancy. Per-task memory and per-run peaks are saved in raw results. The final active payload is measured in each run, not copied from the audit probe.

- **Replay-300 (Research)**: replay_items=300†; replay_capacity=300†; image_bytes=235200†; label_bytes=300†; logit_bytes=0†; buffer_metadata_bytes=16†; buffer_bytes=235516†.
- **Replay-64KiB**: replay_items=83†; replay_capacity=83†; image_bytes=65072†; label_bytes=83†; logit_bytes=0†; buffer_metadata_bytes=16†; buffer_bytes=65171†.
- **DER++-300**: replay_items=300†; replay_capacity=300†; image_bytes=235200†; label_bytes=300†; logit_bytes=12000†; buffer_metadata_bytes=16†; buffer_bytes=247516†.
- **EWC**: fisher_bytes=3256480†; parameter_snapshot_bytes=3256480†; other_cl_array_bytes=0†.

Compare Replay-300, Replay-64KiB and MNEMA using both auxiliary content and total resident arrays. MNEMA has a substantial adaptive/fixed allocation beyond its occupied FastStore rows. All buffers here are bounded; do not describe fixed-capacity replay as unbounded memory growth.

![Accuracy versus memory](plots/accuracy_vs_memory.png)

## 8. Energy / Compute

The original technology cards and instrument are unchanged. Native inference counters are projected using the ASIC 45 nm card. The table reports projection per image over the same evaluation workload. MNEMA native training counts/projections are retained separately in raw records. New baseline training, EWC Fisher estimation and buffer work lack comparable complete instrumentation, so **no cross-method training-energy ranking is reported**. Native counters also omit some arithmetic, use simplifying dtype/traffic assumptions and exclude checkpoint restoration. They are not exhaustive end-to-end energy or evidence of physical hardware efficiency. Wall-clock seconds are measured on this host (including integrity checks), not energy. Raw records report SGD updates, replay draws and Fisher sample counts.

## 9. Favorable debug observations

No research advantage is established by this debug run. Observed debug comparisons only: Final accuracy above Naive MLP by 51.80 percentage points. Final accuracy above Replay-300 (Research) by 5.80 percentage points. Final accuracy above Replay-64KiB by 4.20 percentage points. Final accuracy above EWC by 51.60 percentage points. No significance is implied. Memory and projected-operation observations must be read with the caveats above.

## 10. Unfavorable debug observations

These are debugging observations, not full-data conclusions: Final accuracy at or below DER++-300 by 0.80 percentage points. Its total resident array allocation is 4,391,100 bytes, and its configured FastStore limit is not accurately enforced against actual payload.

## 11. Main Interpretation

The implementation and output pipeline have been exercised, but this quick run cannot answer the full research question. Interpret full-data findings only from a separately identified, validated full-data bundle; this report remains a debug record even when another experiment has completed.

## 12. Limitations

Only MNIST, one fixed task order, simple MLP baselines, limited architecture scale, no validation-based hyperparameter selection, empirical rather than exact model Fisher, task boundaries available to EWC during training, unequal algorithmic compute despite one external pass, simulated/projected energy, no real neuromorphic hardware, array payload rather than peak process RAM. Reproducibility is checked within an environment; exact cross-platform floating-point equivalence is not guaranteed. Reduced samples and one-seed defaults are additional debug limitations.

## 13. Next Experiment

A possible follow-up is Split-FashionMNIST under a separately declared protocol to test whether the trade-off extends beyond MNIST. This repository does not implement that follow-up benchmark.

## Reproduction and artifacts

Full command (stronger computer only): `python experiments/run_research_benchmark.py --mode full --seeds 10`.
Quick command: `python experiments/run_research_benchmark.py --quick`.
Repeat the same command to resume; `--force` archives that mode's previous output directory and reruns all pairs. Interrupted pairs restart from the beginning; completed pairs are durable. Plots can be regenerated using `python experiments/plot_research_benchmark.py results/research_benchmark/quick` and this report using `python experiments/report_research_benchmark.py results/research_benchmark/quick`.

Machine-readable artifacts: [full_results.json](full_results.json), [per_seed_results.json](raw/per_seed_results.json), [summary.json](summaries/summary.json), per-pair raw JSON and exact per-seed train/test indices. Despite its generic filename, full_results.json in a quick directory contains DEBUG data and carries explicit mode metadata.

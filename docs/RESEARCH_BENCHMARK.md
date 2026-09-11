# Research Benchmark Documentation

This document is the operating and reporting record for the full research benchmark. It describes the frozen experiment, the GitHub Actions execution path, the result artifacts, and the supplied completed report bundle.

## Scope and invariants

The benchmark compares the existing MNEMA implementation with five continual-learning baselines on full-data Split-MNIST Class-IL. The cloud work adds execution and collection infrastructure only. It does not modify `mnema/`, model behavior, metric definitions, hyperparameters, historical results, or the benchmark protocol.

The full protocol uses:

- five sequential tasks: `01 -> 23 -> 45 -> 67 -> 89`;
- all 60,000 MNIST training images and all 10,000 test images;
- all ten output classes during inference, with no task identifier or output masking;
- one externally presented training stream per seed;
- seeds `0` through `9`;
- one independent run for each seed-method pair;
- NumPy on CPU with one BLAS thread; no GPU or neuromorphic device;
- Python 3.11.7 and the pinned dependencies in `experiments/research_requirements.txt`.

The unit of replication is the seed. Tables report the sample standard deviation across the ten seeds. The benchmark does not claim statistical significance or select a method by tuning against the final test set.

## Methods

The canonical method slugs used by the runner and workflows are:

| Slug | Reported method |
| --- | --- |
| `naive` | Naive MLP |
| `replay300` | Replay-300 (Research) |
| `replay64kib` | Replay-64KiB |
| `ewc` | EWC |
| `derpp300` | DER++-300 |
| `mnema` | MNEMA |

The runner also accepts `replay64k` as an alias for `replay64kib` and `derpp` as an alias for `derpp300`. The workflow uses canonical slugs.

## GitHub Actions workflows

Both workflows are manual-only. They do not run on pushes and are never launched automatically from the local machine.

### Research Benchmark

Open **Actions -> Research Benchmark -> Run workflow** and select a mode:

| Mode | Jobs | Purpose |
| --- | ---: | --- |
| `smoke/full-seed-0` | 6 | Full dataset, seed 0, all six methods. Use this as the first cloud validation. |
| `full-10-seed` | 60 | Seeds 0-9 for all six methods, followed by aggregation. |

The matrix uses `fail-fast: false`, a maximum of six concurrent jobs, and a 350-minute limit per benchmark job. Each job checks out the repository, installs the pinned research dependencies, obtains the verified MNIST files, and runs exactly one full-data seed-method pair.

### Research Benchmark Pair

Use **Actions -> Research Benchmark Pair -> Run workflow** when one full-data pair is needed. Enter a seed from `0` to `9` and select one canonical method slug. This workflow runs one pair and uploads its output without starting the 60-job matrix.

## Command-line equivalent

The pair runner can be invoked locally or by another controlled runner:

```bash
python experiments/run_research_benchmark.py --mode full --seed 0 --method mnema
```

`--seed` and `--method` must be supplied together. Omitting both preserves the existing all-method runner behavior. Pair-mode output is isolated under:

```text
results/github-actions/seed_<seed>/<method>/
```

The existing all-method output paths under `results/research_benchmark/` are not used by pair mode.

## Data, dependency, and output handling

The preparation job downloads or reuses the four standard MNIST gzip files and verifies their checksums. The prepared configuration and data are uploaded as an input artifact. Matrix jobs download that artifact into their own workspaces, so no job depends on files on a personal computer.

Each pair writes its raw JSON result, worker metadata, and execution log into a unique seed-method directory. The workflow uploads an artifact named:

```text
research-seed-<seed>-<method>
```

Examples include `research-seed-0-mnema` and `research-seed-7-derpp`. Artifacts are retained for 30 days by the workflow; download them before expiry.

## Aggregation and reporting

The `aggregate` job runs only for `full-10-seed`. It waits for the matrix, downloads every pair artifact, reconstructs the result layout expected by the existing collection code, and verifies that all 60 raw results are present. It then runs `experiments/research_actions.py collect`.

The existing reporting code generates:

- `summary.json`;
- `summary_table.md` and `summary.csv`;
- `BENCHMARK_REPORT.md`;
- final-accuracy and forgetting plots;
- accuracy-versus-memory and forgetting-versus-memory plots;
- retention matrices;
- continual-learning progress plots;
- raw results, worker metadata, validation data, and configuration records.

The final bundle is uploaded as `research-benchmark-final`. A failed pair does not cancel other matrix jobs. Review the failed job log and use GitHub's **Re-run failed jobs** action when appropriate.

## Supplied completed benchmark

The completed full-data report bundle supplied with this project is stored outside the repository in the `Desktop\cognx_benchmarking` directory. Its top-level files are:

```text
BENCHMARK_REPORT.md
config.json
full_results.json
status.json
validation.json
plots/
raw/
summaries/
workers/
```

The recorded status is `complete`, with all 60 seed-method artifacts validated. The report records configuration identifier `ab869e57feea62e93198f72fedde18182dc93a28a72d4875e966067d676027a8`, source commit `fe86ec1c6abc600dda8ec50565a551af4e5434bd`, Python `3.11.7`, NumPy `2.2.6`, Matplotlib `3.10.3`, and PyYAML `6.0.2`.

The reported ten-seed means are:

| Method | Final accuracy (%) | Forgetting (percentage points) | Total resident arrays (bytes) | Projected native inference (microjoules/image) |
| --- | ---: | ---: | ---: | ---: |
| Naive MLP | 19.79 +/- 0.06 | 99.52 +/- 0.12 | 814,120 | 1.768 |
| Replay-300 (Research) | 82.44 +/- 1.33 | 20.58 +/- 1.62 | 1,049,636 | 1.768 |
| Replay-64KiB | 67.59 +/- 2.07 | 39.30 +/- 2.60 | 879,291 | 1.768 |
| EWC | 19.98 +/- 0.36 | 99.12 +/- 0.40 | 7,327,080 | 1.768 |
| DER++-300 | 89.39 +/- 0.60 | 12.12 +/- 0.73 | 1,061,636 | 1.768 |
| MNEMA | 77.12 +/- 0.87 | 6.53 +/- 1.41 | 4,391,100 | 0.316 |

These values are reported for the frozen implementation and the stated protocol. Projected inference energy is not measured physical energy. Memory values are resident NumPy array payloads under the report's accounting boundary, not process RSS. The report also documents the FastStore row-size accounting discrepancy and the fact that MNEMA's configured 64 KiB FastStore budget is not a true total-memory ceiling.

## Reproduction and review checklist

Before submitting results, retain the report bundle together with its `config.json`, `validation.json`, raw results, plots, and worker metadata. Confirm that:

1. the run mode is `full`;
2. seeds `0` through `9` are present for all six methods;
3. validation reports all 60 saved artifacts as valid;
4. the configuration identifier and source commit are recorded;
5. the report's caveats are included wherever results are quoted;
6. historical small-benchmark results are kept separate from the full-data research results.

Do not describe projected energy as measured energy, the FastStore configuration as a complete 64 KiB system-memory limit, or the ten-seed means as proof of statistical significance.

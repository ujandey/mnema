# Research benchmark runbook

[Experiment catalog](README.md) · [Protocol and results](../docs/RESEARCH_BENCHMARK.md) · [Troubleshooting](../docs/TROUBLESHOOTING.md)

Run all commands from the repository root using the [Python 3.11.7 research environment](../docs/GETTING_STARTED.md). Select a workload explicitly: omitting both `--mode` and `--quick` selects **full data and ten seeds**.

## Existing full-data evidence

The [completed full bundle](../results/research_benchmark/full/README.md) is included at `results/research_benchmark/full/`: all 60 runs, original reports/plots, raw predictions, validation, and worker metadata. Reading those files requires no training run. An import audit verified the saved artifacts without executing the models.

The import audit recorded one runner content difference and CRLF/LF differences in 18 source files. Later Apache 2.0 license-header updates also change source hashes without changing model behavior. Standard source validation and resume will therefore reject the imported bundle in this checkout. Preserve it as evidence. A fresh current-source experiment must use a separate checkout/output context or deliberately archive it with the all-method `--force` option described below.

## Quick run

```bash
python experiments/run_research_benchmark.py --quick
```

Defaults: seed 0, all six methods, 100 training / 100 test images per task. Output is `results/research_benchmark/quick/`. This is a pipeline check with undefined single-seed SD, not a research comparison. Dataset files are downloaded if missing and verified before use.

## Full run

```bash
python experiments/run_research_benchmark.py --mode full --seeds 10
```

Runs all 60 pairs sequentially and writes `results/research_benchmark/full/`. This path already contains the imported full record, so first address the preservation and source-matching requirements above. The full protocol uses all 60,000 training and 10,000 test images; it can take substantial CPU time, particularly for MNEMA and EWC. Runtime depends on the host. There is no local process or seed parallelism in this command.

## Resume and fresh runs

Repeat the same all-method command to resume. The runner validates completed pairs and skips valid results. An interrupted pair restarts from its beginning; there are no persisted mid-task model checkpoints. Do not run two processes into the same mode directory.

Resume requires matching configuration, source hashes, data, and environment. A mismatch produces `Existing configuration/source/environment differs`. The checked-in debug bundle may differ from a fresh local environment or the current source revision.

To deliberately archive the existing quick bundle and start a new quick experiment:

```bash
python experiments/run_research_benchmark.py --quick --force
```

For a new full experiment:

```bash
python experiments/run_research_benchmark.py --mode full --seeds 10 --force
```

For all-method runs, `--force` moves the selected mode directory to a timestamped sibling such as `quick_archive_<timestamp>` before rerunning every pair. Other modes and historical output files remain in place. Archives are ignored by Git; preserve any evidence you need separately.

## One seed-method pair

```bash
python experiments/run_research_benchmark.py --mode full --seed 0 --method mnema
python experiments/run_research_benchmark.py --mode full --seed 0 --method derpp300
```

Run these independently for the pair you need. `--seed` and `--method` are required together. `--seed` selects an index, whereas `--seeds N` defines the configured range `0,...,N-1`. Full mode defaults to a range of ten seeds; quick mode defaults to a range of one.

Pair output uses `results/github-actions/seed_<seed>/<canonical-slug>/`, even when run locally or with quick data. `derpp` resolves to the canonical `derpp300` directory. Pair runs write configuration, raw result, and completion status; they do not create a complete report, worker metadata, or saved index bundle.

**Pair mode always reruns and replaces that pair's files.** It does not use all-method resume or the `--force` archive branch. Avoid simultaneous runs of the same pair or alternating quick/full pair runs into the same path without preserving earlier results.

## CLI reference

```bash
python experiments/run_research_benchmark.py --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode quick\|full` | `full` | Dataset scope |
| `--quick` | Off | Alias for `--mode quick`; conflicts with `--mode full` |
| `--seeds N` | Quick: 1; full: 10 | Seed range starting at zero |
| `--seed N --method SLUG` | Unset | Run just one pair; seed must be in the configured range |
| `--quick-train N` | 100 | Training images per task in quick mode; 1-500 |
| `--quick-test N` | 100 | Test images per task in quick mode; 1-200 |
| `--lr X` | 0.01 | Positive finite dense-baseline learning rate |
| `--ewc-lambda X` | 100.0 | Nonnegative finite EWC penalty coefficient |
| `--fisher-samples N` | 0 | EWC samples per task; 0 means all; no final-task Fisher |
| `--der-alpha X` | 0.5 | Nonnegative finite logit-matching weight |
| `--der-beta X` | 0.5 | Nonnegative finite replay-label CE weight |
| `--force` | Off | Archive and restart the selected all-method mode directory |

Methods: `naive`, `replay300`, `replay64kib`, `ewc`, `derpp300`, `mnema`. Main-runner aliases: `replay64k`, `derpp`. Changing coefficients, sample limits, or seed coverage defines a different experiment; label it accordingly.

## Validate saved results

```bash
python experiments/validate_research_benchmark.py results/research_benchmark/quick
python experiments/validate_research_benchmark.py results/research_benchmark/quick --reproduce
```

The first command checks saved artifact consistency and writes `validation.json`. The second additionally recomputes quick results. Both require compatible source and MNIST data. Full saved-artifact validation is available without automatic retraining:

```bash
python experiments/validate_research_benchmark.py results/research_benchmark/full
```

`--reproduce` is rejected for full runs. The full validation command above requires a checkout matching the record; it will fail the source check for the imported bundle on the current checkout. Its [IMPORT_VALIDATION.json](../results/research_benchmark/full/IMPORT_VALIDATION.json) documents the separate saved-artifact consistency checks. A validator failure from changed source is a provenance mismatch, not permission to bypass hashes or overwrite the saved configuration.

The [EWC diagnostic](check_ewc_sanity.py) checks the saved quick run's Fisher, snapshots, and actual penalty updates:

```bash
python experiments/check_ewc_sanity.py results/research_benchmark/quick
```

It writes `EWC_SANITY_CHECK.md` and `ewc_sanity.json`. Functional checks do not establish optimal hyperparameters or repair a poor accuracy result.

## Regenerate reports and plots

Create a fresh working copy before regenerating the imported full bundle. The following commands preserve the original files and checksums; the preview directory is ignored by Git and the copy step refuses to overwrite an existing preview:

```bash
python -c "from pathlib import Path; import shutil; preview = Path('.benchmark-actions/report-preview/full'); assert not preview.exists(), 'Preview already exists'; shutil.copytree('results/research_benchmark/full', preview)"
python experiments/plot_research_benchmark.py .benchmark-actions/report-preview/full
python experiments/report_research_benchmark.py .benchmark-actions/report-preview/full
```

Use a separate quick working copy only for debug outputs. Plotting writes PNG/PDF figures. Reporting writes `BENCHMARK_REPORT.md`, summary Markdown/CSV, and `summaries/README_SECTION.md` from saved JSON; it does not retrain models or validate the run for you. These commands replace derived files in the preview directory.

To replace the marked research block in the root README as well:

```bash
python experiments/report_research_benchmark.py .benchmark-actions/report-preview/full --update-readme
```

The root README now presents the completed full-data results. Inspect the resulting diff before retaining it; a quick fragment would replace that primary comparison with debug numbers. Generated README fragments use repository-root-relative links and assume the conventional `results/research_benchmark/<mode>/` layout. For this full-data preview, they point to the preserved full bundle. If you change the underlying data or use another layout, check those links. Keep the start/end markers in the root README intact.

Generated files can contain legacy wording. The curated [artifact guide](../results/README.md) and [protocol](../docs/RESEARCH_BENCHMARK.md) explain the accounting boundaries and corrections; editorial notes added to generated snapshots can be overwritten by regeneration.

## Distributed helper

For controlled workers sharing a prepared configuration, the helper supplies a consistent raw/worker layout. This quick example runs one shard; `collect` requires all six canonical methods for quick mode.

```bash
python experiments/research_actions.py prepare --mode quick --directory .benchmark-actions/prepared-quick
python experiments/research_actions.py run --directory .benchmark-actions/prepared-quick --seed 0 --method naive --output .benchmark-actions/shards-quick/seed_0_naive
```

Run the `run` command for each remaining canonical method, with a distinct output directory per pair. After all six succeed:

```bash
python experiments/research_actions.py collect --directory .benchmark-actions/prepared-quick --shards .benchmark-actions/shards-quick
```

Preparation requires a fresh directory. Each worker needs the same source, pinned software, dataset, and prepared configuration/indices. Full preparation expects all 60 pairs. Duplicate, missing, mismatched, or corrupted results fail collection; do not merge independent configurations.

For the checked-in GitHub workflows, including the current DER++ directory mismatch, see [cloud execution](../docs/RESEARCH_BENCHMARK.md#github-actions).

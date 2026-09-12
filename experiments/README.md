# Experiments and diagnostics

[Project overview](../README.md) · [Research runbook](RESEARCH_BENCHMARK.md) · [Results](../results/README.md)

Run scripts from the repository root using the [research environment](../docs/GETTING_STARTED.md). The research runner, historical experiments, and focused diagnostics answer different questions; use the table below to choose an entry point.

The full-data experiment has already completed. Its [report, summaries, and plots](../results/research_benchmark/full/README.md) are available without running any script. The current source differs from the original run's manifest; consult the provenance notes before attempting reproduction or regeneration.

## Research workflow

| Script | Purpose | Outputs or side effects |
| --- | --- | --- |
| [run_research_benchmark.py](run_research_benchmark.py) | Train/evaluate all methods or one selected pair | Mode-specific report bundle or isolated pair JSON; defaults to full data |
| [validate_research_benchmark.py](validate_research_benchmark.py) | Validate saved predictions, metrics, source, and data | Writes `validation.json`; optional `--reproduce` reruns quick mode only |
| [plot_research_benchmark.py](plot_research_benchmark.py) | Plot saved research results | Six PNG/PDF plot families in the bundle's `plots/` |
| [report_research_benchmark.py](report_research_benchmark.py) | Render saved JSON as report, tables, and README fragment | Replaces derived Markdown/CSV; `--update-readme` also replaces the root README block |
| [check_ewc_sanity.py](check_ewc_sanity.py) | Audit the existing quick EWC configuration and updates | `ewc_sanity.json` and `EWC_SANITY_CHECK.md` inside the quick bundle |
| [research_actions.py](research_actions.py) | Prepare shared inputs, run distributed shards, and collect a complete experiment | Prepared configuration, raw/worker records, validation, final reports |

Start with the [runbook](RESEARCH_BENCHMARK.md) for exact commands. The [protocol](../docs/RESEARCH_BENCHMARK.md) defines dataset order, method differences, metrics, and interpretation.

Supporting modules are not standalone commands:

| Module | Responsibility |
| --- | --- |
| [research_state.py](research_state.py) | Per-image checkpoint restoration, state hashes, and order-independent evaluation |
| [research_memory.py](research_memory.py) | Persistent array inventory and actual FastStore occupancy |
| [research_results.py](research_results.py) | Metrics, digests, atomic writes, result validation, aggregation |

## Dataset-free checks

```bash
python -m unittest discover -s tests -p "test_research*.py" -v
python experiments/e0_calibrate.py
python experiments/e3_separator_test.py
```

| Check | What it establishes | Output |
| --- | --- | --- |
| Research unit tests | Gradients, reservoir budgets, metric rules, evaluation isolation, and distributed integrity on synthetic fixtures | Test log; temporary test artifacts |
| [e0_calibrate.py](e0_calibrate.py) | Counter agreement with the dense-path accounting formula | Console only |
| [e3_separator_test.py](e3_separator_test.py) | Encoder/separator/store write-read diagnostic | Console only; observations can vary with random initialization |

These are software checks, not complete energy validation or dataset accuracy measurements.

## Historical experiments and audits

These scripts retain the older experimental setup and can **overwrite tracked artifacts**. Run them only when you intend to refresh that historical evidence. Their JSON files belong under `results/`, not the research mode directories.

| Command | Question | Output under `results/` |
| --- | --- | --- |
| `python main.py --benchmark` | Small three-method Split-MNIST comparison, seed 0 | `benchmark_results.json` |
| `python experiments/run_split_mnist_benchmark.py 3` | Same historical comparison with an explicit seed (example: 3) | `benchmark_results.json` |
| `python main.py --plot` | Visualize the historical benchmark | `retention_matrices.png`, `energy_accuracy_frontier.png` |
| `python experiments/run_variance.py 10` | Small-benchmark variation over ten seeds | `variance_results.json` |
| `python experiments/run_footprint.py` | Inspect component arrays and per-sample counters | `footprint_results.json` |
| `python experiments/run_energy_breakdown.py` | Attribute native inference projection to counter terms | `energy_breakdown.json` |
| `python experiments/run_fewshot.py 3` | New-class acquisition across three seeds | `fewshot_results.json` |
| `python experiments/run_code_geometry.py` | Compare sparse-code overlaps | `code_geometry.json` |
| `python experiments/run_uncounted_audit.py` | Inspect eligibility and synaptic arithmetic accounting | `uncounted_audit.json` |
| `python experiments/experiments_test.py` | Older integrated model diagnostic using MNIST | Console only |
| `python experiments/export_tables.py` | Assemble historical JSON and embedded descriptive material into tables | `ALL_TABLES.md`, `tables/*.csv` |

Historical footprint and table descriptions contain superseded memory categories and claims. Read the [historical artifact notes](../results/README.md#historical-artifacts); the research array inventory is the current accounting reference. Exporting tables does not retrain models, validate all claims, or turn historical results into full-data evidence.

## Reproducibility habits

Keep configuration, raw predictions, indices, validation, and derived summaries together. Record changes to seeds, coefficients, source, or environment. Do not edit JSON to improve a score, remove an inconvenient baseline, or relabel a quick run as full. Preserve failed-run logs when they explain missing coverage.

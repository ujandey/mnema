# Results and artifact guide

[Project overview](../README.md) · [Research protocol](../docs/RESEARCH_BENCHMARK.md) · [Runbook](../experiments/RESEARCH_BENCHMARK.md)

This directory contains **research debug artifacts and historical experiments**. It does not contain the separately supplied full-data ten-seed bundle. The [recorded full-data results](../docs/RESEARCH_BENCHMARK.md#recorded-full-data-results) identify that bundle and its configuration.

## Choose the correct evidence

| Family | Scope | Primary entry point |
| --- | --- | --- |
| Research quick | Six methods; seed 0; 100 train / 100 test images per task; isolated checkpoint evaluation | [Debug report](research_benchmark/quick/BENCHMARK_REPORT.md) |
| Historical benchmark | Three methods; 80 train / 50 test images per task; older evaluation semantics | [benchmark_results.json](benchmark_results.json) and [historical tables](ALL_TABLES.md) |
| Historical diagnostics | Focused footprint, geometry, acquisition, variability, and counter probes | Individual JSON files below |

`full_results.json` is a generic bundle filename. In the quick directory it contains reduced debug results. Read `config.mode`, seeds, sample counts, and `research_complete`; never infer experimental scope from a filename or completion status alone.

## Research bundle layout

Paths below are relative to a completed mode directory such as `research_benchmark/quick/`.

| Path | Contents | How to use it |
| --- | --- | --- |
| `config.json` | Protocol, hyperparameters, source/data hashes, software, configuration ID | Establish exactly what ran |
| `status.json` | Completion and research-completion flags | Check status alongside pair coverage |
| `raw/seed_<n>_<slug>.json` | One method's metrics, predictions, counters, memory, and integrity records | Audit a single result |
| `raw/seed_<n>_indices.json` | Exact train/test indices for each task | Recover stream and test selection |
| `raw/per_seed_results.json` | All expected pair records | Input for validation and aggregation |
| `full_results.json` | Configuration plus combined runs | Self-contained result/configuration view |
| `summaries/summary.json` | Aggregate metrics, SD, memory, coverage | Source for derived tables |
| `summaries/summary_table.md`, `summary.csv` | Human-readable and spreadsheet tables | Read values with mode and units intact |
| `summaries/README_SECTION.md` | Generated root-README fragment | Intended for embedding; its links are relative to the repository root |
| `BENCHMARK_REPORT.md` | Generated explanation and figures | Read together with configuration and caveats |
| `plots/` | PNG/PDF accuracy, forgetting, memory trade-offs, retention, progress | Keep the experiment label with exported figures |
| `validation.json` | Saved-artifact checks, optionally quick reproduction | Distinguish checking artifacts from rerunning training |
| `workers/` | Distributed-worker provenance, when collected through that path | Trace the origin of a distributed result |

The quick bundle additionally contains [EWC_SANITY_CHECK.md](research_benchmark/quick/EWC_SANITY_CHECK.md), [ewc_sanity.json](research_benchmark/quick/ewc_sanity.json), and a development validation record. These document a diagnostic run, not a new tuned EWC result.

## Read numbers correctly

- **Accuracy:** percentage, macro-averaged across task test sets.
- **Forgetting:** percentage points, with the exact pre-final maximum defined in the protocol; negative values are valid.
- **Uncertainty:** sample SD across seeds, undefined for a one-seed result.
- **Memory:** unique resident NumPy array payload. Add adaptive, fixed, and allocated auxiliary categories; do not also add active auxiliary content.
- **Energy:** partial native operation-count projection, not measured device energy. Do not rank complete training energy from research outputs.

MNEMA's configured FastStore limit counts 25 bytes per row instead of the actual 28-byte default payload. It excludes other model state and is not a physical 64 KiB ceiling. The [accounting reference](../docs/RESEARCH_BENCHMARK.md#memory-and-energy-accounting) details exclusions.

## Historical artifacts

| Artifact | Source experiment |
| --- | --- |
| [benchmark_results.json](benchmark_results.json) | Small three-method seed-0 benchmark |
| [variance_results.json](variance_results.json) | Ten-seed repetition of the small historical setup |
| [footprint_results.json](footprint_results.json) | Historical component/operation probe; category labels are superseded |
| [energy_breakdown.json](energy_breakdown.json) | Native inference projection attribution |
| [code_geometry.json](code_geometry.json) | Measured code overlap under selected separator settings |
| [fewshot_results.json](fewshot_results.json) | Historical new-class acquisition probe |
| [uncounted_audit.json](uncounted_audit.json) | Audit of dense eligibility and synaptic arithmetic |
| [ALL_TABLES.md](ALL_TABLES.md), [tables/](tables/) | Exported historical tables and CSV snapshots |

Historical evaluation could change MNEMA state through held-out examples. It also used different replay ordering, sample selection, retention coverage, and forgetting rules. Historical accuracy is not directly comparable to the research protocol.

Historical tables/CSVs include obsolete descriptions of cortex state as fixed, replay as unbounded, FastStore as a strict 64 KiB ceiling, and sparse codes as a privacy guarantee. The curated Markdown table notes correct these interpretations while retaining numerical observations; CSV snapshots preserve the old export. Absence of raw-image replay does not establish non-invertibility or privacy. Idealized radio/battery estimates are illustrative assumptions, not measured battery life or cloud cost.

## Regeneration and preservation

Use the [runbook](../experiments/RESEARCH_BENCHMARK.md#regenerate-reports-and-plots) to regenerate a research bundle. Plot/report generators replace derived files. `export_tables.py` replaces historical Markdown/CSV and can restore superseded prose: it remains unchanged as part of the historical source record. Editorial context in generated snapshots may be overwritten, so retain this guide and the research protocol alongside exports.

Do not hand-edit raw results, digests, predictions, or measured values. Preserve complete bundles, including validation and source identifiers. A local all-method `--force` run archives its old mode directory; individual pair commands overwrite their pair files directly. Publication and redistribution remain subject to the [license](../LICENSE).

# Troubleshooting

[Documentation index](README.md) · [Setup](GETTING_STARTED.md) · [Runbook](../experiments/RESEARCH_BENCHMARK.md)

Start from the repository root. Capture the exact command and error before changing an environment or replacing saved results.

## Environment and imports

| Symptom | Cause and next step |
| --- | --- |
| Python 3.11 fails `pip install .` | Package metadata requires >=3.14. For research, install `experiments/research_requirements.txt` and run scripts directly. |
| `uv sync` selects Python 3.14 | `.python-version` and `pyproject.toml` define the package environment. Use the standalone research environment for pinned benchmark reproduction. |
| `ModuleNotFoundError` for NumPy, Matplotlib, or YAML | Install research requirements using the same interpreter that runs the script: `python -m pip install -r experiments/research_requirements.txt`. |
| `mnema-arch` only prints a greeting | The installed console entry point is a placeholder. Use `python main.py` or the research runner directly. |
| PowerShell rejects `Activate.ps1` | Invoke `.\.venv\Scripts\python.exe` directly; activation is optional. |
| Technology card not found | The path is resolved from the working directory. Change to the repository root. |

Inspect the actual interpreter and package versions:

```bash
python -c "import sys, numpy, matplotlib, yaml; print(sys.executable); print(sys.version); print(numpy.__version__, matplotlib.__version__, yaml.__version__)"
```

## Dataset download or checksum failure

The loader expects these original gzip files under `data/`:

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

Missing files are downloaded through [download_mnist](../benchmarks/split_mnist.py). A restricted network may require obtaining the same original files through an approved connection and placing them in `data/`; leave them compressed.

`MNIST checksum mismatch` usually means a truncated, changed, or incorrect file. Preserve the failing file if needed for diagnosis, replace that specific file with the original dataset file, and rerun verification. Do not disable checksum checks. Quick mode still loads and verifies the complete dataset.

## Existing configuration or source differs

For `Existing configuration/source/environment differs`, compare the saved `config.json` with the current source, Python/dependency versions, dataset, and CLI settings. This guard prevents an invalid mixture of runs.

To continue an old experiment, use its matching checkout/environment. To start a new one, deliberately archive with the documented all-method `--force` command. Do not hand-edit `config_id` or source hashes. See [resume and fresh runs](../experiments/RESEARCH_BENCHMARK.md#resume-and-fresh-runs).

The validator's `Source changed since this run` has the same provenance implication. It can occur with the checked-in quick bundle if the current checkout differs from its recorded manifest. This is separate from a numerical test failure.

For the imported full bundle, the current runner has a different content hash and 18 other files differ only in line endings. Its saved-artifact consistency audit passed, but exact-source validation on this checkout will reject it. Read [the provenance notes](../results/research_benchmark/full/README.md#provenance-and-reproduction); do not change original hashes to make validation pass.

## A run seems unexpectedly large

The research runner defaults to full mode with ten seeds. Use `--quick` explicitly for development. `--seeds 1` means one full-data seed unless quick mode is also selected. A seed-0 cloud smoke run likewise uses full data.

EWC Fisher estimation and MNEMA consolidation can add substantial work. All-method execution is sequential; inspect the current task/pair log before assuming it is stuck. Completed pairs are durable, but an interrupted pair must restart.

## Missing reports, plots, or aggregation

| Symptom | Explanation and next step |
| --- | --- |
| A pair directory has no report | Main-runner pair mode writes raw/config/status only. Use an all-method bundle or the distributed prepare/run/collect path for complete reports. |
| Full report or plots missing | The complete bundle belongs under `results/research_benchmark/full/`. Verify that the checkout includes all imported artifacts listed in [IMPORT_MANIFEST.json](../results/research_benchmark/full/IMPORT_MANIFEST.json); do not substitute quick figures. |
| Collector reports missing or duplicate results | It requires exactly the configured seed-method set and matching worker records. Restore the missing shard or remove accidental duplicate inputs from the collection set after inspecting them. |
| GitHub DER++ job cannot copy its raw file | The checked-in YAML looks in `derpp/`, while the runner writes `derpp300/`. See the [known workflow mismatch](RESEARCH_BENCHMARK.md#github-actions). |
| Full matrix has no final artifact after a failed job | Aggregation requires successful prerequisites and all expected results. Diagnose the failed pair before retrying. |
| README fragment images do not render inside `summaries/` | The fragment is designed for embedding in the root README. Open the sibling report or apply it through the documented generator command. |
| A regenerated table lost editorial context | Generated files are replaced by scripts. Keep the [result guide](../results/README.md) alongside exports. |

## Understanding surprising numbers

- A one-seed SD shown as undefined is correct, not missing computation.
- Low forgetting alongside low accuracy can reflect poor acquisition rather than good retention.
- Native inference can change predictions over time without label-driven learning; use checkpoint-isolated evaluation for the research protocol.
- A configured 64 KiB FastStore can exceed that active payload and allocate much more total memory.
- A low projected microjoule value does not establish low host CPU energy, measured hardware energy, or battery life.

See [architecture](ARCHITECTURE.md) and [metric/accounting definitions](RESEARCH_BENCHMARK.md) for the underlying behavior.

## Useful information for a maintainer

Provide the command, working directory, source commit, Python/dependency versions, traceback or failed test name, mode, and affected artifact's configuration ID. Explain whether you were resuming, validating, regenerating, or starting a fresh run. Share proprietary source and result bundles only through an authorized channel.

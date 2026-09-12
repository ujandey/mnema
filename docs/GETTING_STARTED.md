# Getting started

[Documentation index](README.md) · [Troubleshooting](TROUBLESHOOTING.md)

This guide takes an authorized developer from a checkout to a verified local research environment. All commands assume the repository root as the working directory.

## Choose an environment

The repository currently has two dependency definitions. Keep them separate.

| Environment | Python | Dependencies | Intended use |
| --- | --- | --- | --- |
| Research | Tested with **3.11.7** | NumPy 2.2.6, Matplotlib 3.10.3, PyYAML 6.0.2, pinned in [research_requirements.txt](../experiments/research_requirements.txt) | Benchmark reproduction, unit tests, non-GUI diagnostics |
| Package/demo | **3.14**, selected by [.python-version](../.python-version); metadata requires >=3.14 | [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock), including OpenCV and Streamlit | Existing `uv` workflow and interactive demos |

For comparisons with the supplied full-data record, use the research environment. `uv sync` follows the package metadata; it does not install the research pins. A Python 3.11 environment cannot install the project as a package under the current metadata, but can run its scripts directly from the checkout.

No PyTorch, CUDA, Brian2, pretrained checkpoint, or neuromorphic board is needed. Dataset access requires the four MNIST gzip files or network access to download them. The tests and calibration below work without MNIST.

## Get the source

```bash
git clone https://github.com/ujandey/COGNX_2.git
cd COGNX_2
```

If you already have an authorized checkout, use it directly. Repository visibility does not replace the [license terms](../LICENSE).

## Set up research dependencies

### Windows PowerShell

Use an installed Python 3.11.7 interpreter. The launcher selects the installed 3.11 version, so verify its patch version first.

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r experiments/research_requirements.txt
```

If PowerShell blocks activation, use the environment's interpreter directly without changing execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments/research_requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_research*.py" -v
```

Use that same interpreter path in place of `python` in subsequent commands.

### Linux or macOS

Use an installed Python 3.11.7 interpreter:

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/research_requirements.txt
```

### Confirm versions

```bash
python -c "import sys, numpy, matplotlib, yaml; print(sys.version); print('NumPy', numpy.__version__); print('Matplotlib', matplotlib.__version__); print('PyYAML', yaml.__version__)"
```

Expected research versions: Python 3.11.7, NumPy 2.2.6, Matplotlib 3.10.3, PyYAML 6.0.2. Saved configurations record these versions and source hashes. Identical dependency versions are necessary for the prescribed environment, but do not guarantee identical floating-point results on every platform.

## Verify without downloading data

```bash
python -m unittest discover -s tests -p "test_research*.py" -v
python experiments/e0_calibrate.py
python experiments/e3_separator_test.py
```

The unit tests exercise protocol and artifact integrity with synthetic fixtures. Calibration should report 406,528 counted MACs per sample, matching its dense-path accounting formula. The separator diagnostic exercises encoding, sparse selection, and associative write/read. These checks validate specific software paths; they do not measure physical energy or generalization.

## Run the debug benchmark

```bash
python experiments/run_research_benchmark.py --quick
```

Quick mode uses seed 0, all six methods, and 100 training / 100 test images per task. It still loads the complete MNIST dataset as uint8 arrays before selecting subsets. The shared image payload is about 55 MB; that is not a process RAM requirement or peak-memory bound.

Successful completion writes `results/research_benchmark/quick/` with a `complete` status, six raw results, summary tables, figures, and a report. A one-seed debug run has undefined sample standard deviation and is not research evidence.

This repository includes an existing quick bundle. If the runner refuses to resume it, consult [resume and fresh runs](../experiments/RESEARCH_BENCHMARK.md#resume-and-fresh-runs). The documented `--force` option archives the old mode directory before starting again; do not remove integrity checks to reuse mismatched results.

## Validate a compatible saved run

```bash
python experiments/validate_research_benchmark.py results/research_benchmark/quick
```

Validation checks the saved source manifest, dataset, predictions, metrics, and summaries, then writes `validation.json`. It may download MNIST if the dataset is absent. With `--reproduce`, quick validation also reruns the experiment and checks reproducibility. Full-data automatic reruns are rejected.

## Set up the interactive demos

In a separate checkout/environment using the repository's `uv` configuration:

```bash
uv sync --locked
uv run python main.py --demo
uv run streamlit run demo/dashboard.py
```

Run the last two commands separately. The webcam needs a graphical desktop and camera access. The dashboard reads historical saved results; it is not connected to the webcam or research runner. See [demo controls and limitations](../demo/README.md).

## Next steps

- [Use MNEMA from Python](API.md) for a dataset-free example.
- [Run the full protocol](../experiments/RESEARCH_BENCHMARK.md#full-run) after checking resources and output provenance.
- [Read the architecture](ARCHITECTURE.md) before modifying model behavior.

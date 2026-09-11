# Research benchmark operation

MNEMA and the historical benchmark are frozen. Development validation uses only
quick mode. The completed full-data report and cloud execution guide are documented in [docs/RESEARCH_BENCHMARK.md](../docs/RESEARCH_BENCHMARK.md).

## Setup on the stronger computer

Use Python 3.11 and the tested standalone dependency pins (the historical
`pyproject.toml` currently requests Python 3.14 and newer dependency versions):

```bash
python -m pip install -r experiments/research_requirements.txt
python -m unittest discover -s tests -p test_research_benchmark.py -v
```

Run from the repository root. No PyTorch or additional plotting dependencies are
needed. Original MNIST gzip files in `data/` are reused and checked against their
standard checksums; missing files are downloaded using the existing loader helper.

## Development only

```bash
python experiments/run_research_benchmark.py --quick
python experiments/validate_research_benchmark.py results/research_benchmark/quick --reproduce
```

Defaults: one seed, 100 training and 100 test images per task, all six methods,
unchanged MNEMA dimensions. Quick results are explicitly DEBUG ONLY; one-seed SD
is undefined. Models and seeds run sequentially. Images occupy about 55 MB as
uint8 arrays; no full normalized dataset copies or per-image model deepcopies.

## Full experiment — stronger computer only

```bash
python experiments/run_research_benchmark.py --mode full --seeds 10
```

Output: `results/research_benchmark/full/`. Repeat the exact command to resume.
Every completed seed × method result is atomically persisted and validated before
reuse. An interrupted pair restarts; model state is not saved mid-task. There is
no concurrent process/seed execution; do not run multiple runners into one output
directory. Source, settings, dataset and software versions must match on resume.

To archive the previous full directory and start all pairs again:

```bash
python experiments/run_research_benchmark.py --mode full --seeds 10 --force
```

`--force` only moves the selected mode directory to a timestamped sibling inside
`results/research_benchmark/`; historical outputs and the other mode are untouched.
Full data is the default without `--quick`, so explicitly select quick mode on
the development machine.

EWC defaults to lambda=100 and full empirical-Fisher estimation over each of the
first four task training sets, with no weight updates during estimation. Use
`--fisher-samples N` for a declared uniform subset if compute requires it. This is
an extra training-data read, not an extra optimization epoch. DER++ defaults to
alpha=beta=0.5. These settings are fixed, not tuned against the final test set.

## Regenerate artifacts

```bash
python experiments/plot_research_benchmark.py results/research_benchmark/full
python experiments/report_research_benchmark.py results/research_benchmark/full --update-readme
python experiments/validate_research_benchmark.py results/research_benchmark/full
```

The report generator creates the table and README fragment directly from JSON;
`--update-readme` explicitly installs that generated section. Use the quick path
for debug artifact regeneration. Validation never automatically repeats full
runs, even when `--reproduce` is requested (it rejects that combination).

## Interpretation and caveats

- Five tasks: 01 → 23 → 45 → 67 → 89; all 60,000 training and 10,000 test images
  in full mode. Every task stream is shuffled once per seed and shared by methods.
- All predictions cover 0–9; test labels are used only after prediction for scoring.
- Each test image starts at the same post-training MNEMA checkpoint, including
  its membrane state. Mutable inference state is restored in a `finally` block.
  Full attribute and RNG hashes check isolation; reversed test-order predictions
  must agree image by image. No model source changes implement this protocol.
- Every retention cell is measured. Forgetting for old task j uses its best
  post-learning, pre-final score minus its final score; negative values are valid.
- Reservoir Replay-300 (Research) differs intentionally from historical random-
  replacement replay. The historical baseline module remains unchanged.
- Replay-64KiB stores raw uint8 pixels and includes buffer counters in its strict
  65,536-byte array-payload budget. It is not exactly memory matched to MNEMA.
- FastStore counts 25 bytes/row while its active array payload uses 28. Actual
  usage, exceedances and post-sample peaks are reported without fixing eviction.
- Model/adaptive, frozen scaffold, active auxiliary content, allocated auxiliary
  arrays and total resident array payload are distinct. Allocated capacity and
  active content are not added twice. Python/runtime/PRNG overhead, dataset and
  scratch buffers are outside the declared array-payload accounting boundary.
- Native projected inference energy is partial, not measured physical energy.
  New baseline training has no comparable energy instrumentation. Do not rank
  complete training energy using these results.

## Audit record

Historical experiment: 80 training / 50 test examples per task, one unshuffled
pass, ten output classes, Naive/Replay/MNEMA, seed 0 with separate ten-seed small
variance script. Historical evaluation mutated MNEMA state; FastStore undercounted
payload; missing future retention cells were encoded as zero; forgetting included
the final checkpoint in its maximum. Historical source and results are preserved.
The new protocol does not support direct before/after accuracy attribution to
MNEMA, since data volume, ordering and evaluation semantics differ.

# MNEMA: Memory-Native Event-Driven Architecture for Edge Continual Learning

<!-- RESEARCH_BENCHMARK_START -->
## Research Benchmark

**DEBUG RESULTS ONLY — reduced data, not research evidence. The full ten-seed experiment has NOT been completed by this quick run.**

† One seed: standard deviation is undefined; values are single-seed observations.

Full Split-MNIST, five sequential digit pairs, one external training pass, ten-class inference without task IDs; default full seeds 0–9. Methods: Naive MLP, reservoir Replay-300 (Research), Replay-64KiB, EWC, DER++-300 and frozen MNEMA.

Run on a stronger machine: `python experiments/run_research_benchmark.py --mode full --seeds 10`. Resume with the same command; add `--force` to archive and rerun. Development: `python experiments/run_research_benchmark.py --quick`.

| Method | Final ACC (%) | Forgetting (pp) | Model/adaptive (B) | Fixed scaffold (B) | Auxiliary content (B) | Auxiliary allocated (B) | Total resident arrays (B) | Native projected inference (µJ/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive MLP | 19.40† | 98.25† | 814120† | 0† | 0† | 0† | 814120† | 1.768† |
| Replay-300 (Research) | 65.40† | 37.50† | 814120† | 0† | 235516† | 235516† | 1049636† | 1.768† |
| Replay-64KiB | 67.00† | 35.00† | 814120† | 0† | 65171† | 65171† | 879291† | 1.768† |
| EWC | 19.60† | 98.00† | 814120† | 0† | 6512960† | 6512960† | 7327080† | 1.768† |
| DER++-300 | 72.00† | 26.75† | 814120† | 0† | 247516† | 247516† | 1061636† | 1.768† |
| MNEMA | 71.20† | 24.25† | 3408032† | 524316† | 73388† | 458752† | 4391100† | 0.316† |

Memory columns are array payload; total = adaptive + fixed + allocated auxiliary. Native inference energy is an incomplete projection, not measured energy. MNEMA's configured 64 KiB limit is not an actual payload ceiling (25 bytes counted versus 28 bytes per active row). Stateless checkpoint evaluation is enforced per image. These results are not comparable directly with historical small-benchmark numbers.

[Detailed report](results/research_benchmark/quick/BENCHMARK_REPORT.md) · [Generated summary](results/research_benchmark/quick/summaries/summary.json)

![Accuracy](results/research_benchmark/quick/plots/final_accuracy.png)
![Memory trade-off](results/research_benchmark/quick/plots/accuracy_vs_memory.png)
![Retention](results/research_benchmark/quick/plots/retention_matrices.png)

<!-- RESEARCH_BENCHMARK_END -->

### Run the research benchmark on GitHub Actions

The cloud benchmark is manual-only. Open [Actions -> Research Benchmark](https://github.com/ujandey/COGNX_2/actions/workflows/research-benchmark.yml), select **Run workflow**, choose branch **main**, and select one of these modes:

- **`smoke/full-seed-0`**: full data for seed 0 across all six methods;
- **`full-10-seed`**: seeds 0-9 across all six methods, followed by aggregation.

For one full-data seed-method pair, use [Actions -> Research Benchmark Pair](https://github.com/ujandey/COGNX_2/actions/workflows/research-benchmark-pair.yml). The workflow uses the pinned research environment, verified MNIST files, isolated output directories, and independent CPU jobs. It does not run on ordinary pushes.

Each pair uploads a `research-seed-<seed>-<method>` artifact. The full matrix also uploads `research-benchmark-final`, containing the validated results, tables, plots, and report. See [`docs/RESEARCH_BENCHMARK.md`](docs/RESEARCH_BENCHMARK.md) for the complete protocol, artifact layout, reporting caveats, and the supplied full-data benchmark record.

> **A biologically grounded, event-driven continual learning framework that bounds memory growth, eliminates dense backbone pre-training energy, and operates via sparse address-and-accumulate synaptic dynamics.**

> ⚠️ **PROPRIETARY AND CONFIDENTIAL — COPYRIGHT © 2026 COGNX. ALL RIGHTS RESERVED.**
> This is **not** open-source software. No right to use, copy, modify, or distribute is
> granted. Authorized recipients may **view** the source solely to evaluate COGNX. See
> [`LICENSE`](LICENSE) for the full terms, including restrictions on publishing benchmark
> results derived from this work.

---

## ⚡ Core Architectural Principles

1. **Plasticity in the Representation:** No frozen GPU backbones. Front-end converts raw streams into temporal contrast / TTFS spike rasters ($[E]$).
2. **Sparse Orthogonalization ($[S]$):** Random sparse expansion ($N_{\text{in}} \rightarrow N_S = 16,384, c=8$) with $k$-WTA ($k=64$). Expected inter-class code collision is $\approx \frac{k^2}{N_S} = 0.25$ units, eliminating catastrophic interference by construction.
3. **Associative Store ($[F]$):** Address-and-accumulate inference with a configured occupancy budget $B$ and salience-based eviction. The current byte accounting undercounts actual payload; see the research benchmark caveats.
4. **Metaplastic Slow Cortex ($[C]$):** Adaptive Leaky Integrate-and-Fire (ALIF) network paired with 4-variable ($m=4$) Benna–Fusi complex synapses, yielding power-law memory retention ($\sim \sqrt{t}$).
5. **Sparse-Code Sleep Replay ($[Z]$):** Replays stored 112-byte sparse indices directly into Cortex without re-running the front-end encoder, recycling fast-store capacity upon consolidation.
6. **Audited Energy Instrument ($[X]$):** Every operation is intercepted and logged into real hardware op-counts, then projected against published technology cards (Intel Loihi 2, 45nm ASIC, 28nm ODIN).

---

## 📊 Historical Small Benchmark (Split-MNIST 5-Task)

Historical debugging experiment: **80 training / 50 test examples per task**, seed 0.
Its evaluation allowed test inputs to alter MNEMA state. These results are preserved
for historical context and are not full-data research results. Reproduce with
`uv run python main.py --benchmark` (this overwrites the historical output).
Source of truth: `results/benchmark_results.json`.

| Architecture | Final Acc ($\text{ACC}$) | Forgetting ($\text{FM}$) | Dense MACs | Sparse SynOps | Projected Energy ($45\text{nm}$ ASIC) | Learned-Memory Growth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP (Naive Backprop)** | 19.2% | 92.0% | 315.1M | 0 | 2,334 µJ | Bounded (Dense) |
| **MLP (+ Replay Buffer)** | **62.8%** | 32.0% | 477.7M | 0 | 3,343 µJ | Linear $O(N)$ — 919 KB buffer |
| **MNEMA (Historical)** | 57.6% | **23.0%** | 1,624.6M | 741.8k | 10,347 µJ | Configured 64 KiB; actual payload undercounted |

Over 10 seeds: Naive `19.0 ± 0.6%`, Replay `63.1 ± 3.5%`, MNEMA `56.2 ± 4.6%`
(`uv run python experiments/run_variance.py 10`).

### Historical observations (not full research conclusions)

MNEMA's advantage is on the **inference path**, and it is architectural:

| Per single inference | MNEMA | Dense MLP |
| :--- | :---: | :---: |
| Dense MACs | **0** | 203,264 |
| SRAM bytes touched | **243,651** | 813,056 |
| Projected energy (45nm) | **0.316 µJ** | 1.768 µJ (**5.59×**) |
| Accuracy per µJ | **182.0** | 35.5 |

MNEMA had the lowest forgetting in this historical three-method experiment.
Its configured FastStore budget is 64 KiB, but the implementation counts 25 bytes
per occupied row where the actual NumPy row payload is 28 bytes. Cortex and other
adaptive state are additional memory. The implementation does not store raw-image replay.

MNEMA is currently **behind** on three axes, and the repository measures all of them:

- **Accuracy:** replay beats MNEMA by ~6 points; the gap exceeds the error bars.
- **Training energy:** the Benna–Fusi cortex update costs ~1.64M dense MACs per step, making
  the historical small benchmark ~3× more expensive than replay under its projection.
- **Historical memory estimates:** superseded for research comparisons by direct array
  inventories in the new report. The 64 KiB configuration applies only to FastStore's
  undercounted occupancy; MNEMA also contains substantial trainable/adaptive state.

### Honest energy accounting

The `[X]` instrument charges every operation, **including MNEMA's own dense paths**. An earlier
revision left the cortex eligibility-trace decay (163,840 fp32 multiplies per forward) and the
Benna–Fusi diffusion (~1.64M multiplies per update) uncounted, understating projected energy by
~26× on the historical small benchmark. Both are now instrumented, and the eligibility trace is gated behind
`is_training` so inference neither pays for it nor mutates plasticity state.

Verify with `uv run python experiments/run_uncounted_audit.py`.

---

## 🚀 Quickstart

### 1. Installation & Environment
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/MNEMA.git](https://github.com/YOUR_USERNAME/MNEMA.git)
cd MNEMA

# Install dependencies via uv
uv sync
```

### 2. Run the benchmark and demos
```bash
uv run python main.py --benchmark   # Split-MNIST Class-IL, seeded (default 0)
uv run python main.py --plot        # Retention matrices + energy/accuracy frontier
uv run python main.py --demo        # Live webcam edge demonstrator
uv run streamlit run demo/dashboard.py   # Telemetry dashboard
```

### 3. Reproduce the full evidence set
```bash
uv run python experiments/e0_calibrate.py         # Instrument calibration (0.0% discrepancy)
uv run python experiments/e3_separator_test.py    # [E]->[S]->[F] pipeline sanity check
uv run python experiments/run_variance.py 10      # 10-seed error bars
uv run python experiments/run_footprint.py        # Byte-exact memory + per-sample cost
uv run python experiments/run_energy_breakdown.py # Compute vs data-movement attribution
uv run python experiments/run_fewshot.py 3        # New-class acquisition curves
uv run python experiments/run_code_geometry.py    # Intra/inter-class code overlap frontier
uv run python experiments/run_uncounted_audit.py  # Self-audit for uncharged arithmetic
```

Each writes a JSON artifact into `results/`.

---

## 🔬 Scope & Honest Claims

| Claim | Status |
| :--- | :--- |
| Zero dense MACs on the **inference** path | ✅ Architectural, instrument-verified |
| Configured FastStore budget (64 KiB) | Actual active payload is undercounted; not a true 64 KiB ceiling |
| Lowest forgetting vs both baselines | ✅ At seed 0; error bars overlap at n=10 |
| Sparse orthogonal codes | ✅ Measured: 2.00 inter-class overlap of k=64 |
| Runs on neuromorphic hardware | ❌ NumPy/CPU; energy is **projected**, not measured |
| Brian2 simulation | ❌ Not a dependency |
| True event-camera input | ❌ Webcam frame-diff shown for display; intensity fed to model |
| Full multi-layer SNN | ❌ Spike encoder + ALIF output layer only; `hidden_dim` unused |
| Category-level one-shot learning | ❌ ~50 shots for a new class; intra-class code overlap is only 9.3% |

`project_energy()` returns `is_measured: False` on every call by design.

---

## ⚖️ License

**Copyright © 2026 COGNX. All Rights Reserved.**

This software is **proprietary and confidential**. It is not open source, not free
software, and not in the public domain.

| | |
| :--- | :--- |
| **May view** | Authorized recipients, solely to evaluate COGNX |
| **May not** | Use, run, copy, fork, modify, distribute, sublicense, or sell |
| **May not** | Reverse engineer the architecture or algorithms |
| **May not** | Use as training data or reference for any AI/ML system or competing product |
| **May not** | Publish benchmark results or energy projections derived from this work without written consent |
| **May not** | Use the COGNX or MNEMA names or marks |

Full terms in [`LICENSE`](LICENSE). Third-party dependencies (NumPy, OpenCV, Matplotlib,
PyYAML, Streamlit) remain under their own licenses and are not covered by these terms.

For licensing, evaluation access, or commercial enquiries, see the contact section of the
[`LICENSE`](LICENSE) file.

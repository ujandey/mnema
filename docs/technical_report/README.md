# COGNX Technical Report 001

**MNEMA: Exploring Fast Associative Memory and Slow Adaptive Learning for Continual Learning**  
Ujan Dey and Swapnil · COGNX Research · Version 0.1 · September 2026

This is a technical report, **not a peer-reviewed paper**. All MNEMA results come from the repository's completed full benchmark. Preparing the report did not rerun training or inference, change research code, or modify benchmark data.

## Files and layout

- [Compiled report](../mnema.pdf): the reader-ready PDF linked from the repository and documentation front pages.
- [main.tex](main.tex): manuscript, equations, three tables, and an inline TikZ architecture diagram.
- [references.bib](references.bib): four original academic references and two repository references.
- [figures/retention_matrices.pdf](figures/retention_matrices.pdf): unchanged full-benchmark retention figure.
- [figures/accuracy_vs_memory.pdf](figures/accuracy_vs_memory.pdf): unchanged full-benchmark auxiliary-memory figure.
- [README.md](README.md): build instructions, evidence map, audit decisions, and validation status.

The manuscript uses one column, 10-point Latin Modern type, 23 mm margins, black body text, numbered sections, and page numbers after the title page. The reader-reviewed build had **13 A4 pages, including the title page and references**; that length is acceptable. Explicit page breaks group the major topics. The revised protocol uses tighter local paragraph/display spacing and condensed prose to keep its metrics on page 5 and start Results on page 6. The original scientific plots retain their source colors. The architecture is a vector TikZ drawing, not an AI-generated illustration.

The revised pagination has **not been compiled locally**. Source checks cannot establish the new page count or rule out typesetting warnings. On recompilation, verify that Section 4 starts on page 6 and that no short metrics-only page remains.

The cover now identifies the series once as “COGNX Research · Technical Report 001”, with COGNX Research retained as the authors' affiliation. Figure 2 and its caption are unchanged. Figure 3 has a small note directly above the chart, “x-axis is not total resident memory”, inside the same float so the note stays with the plot. Appendix A keeps the benchmark command, recorded environment and source commit, and the essential reproduction boundary; detailed provenance remains in this README.

## Compile locally or in Overleaf

Run from this directory with a standard TeX Live or MiKTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload `main.tex`, `references.bib`, and the `figures/` directory together. Select `main.tex` as the main document and pdfLaTeX as the compiler. Overleaf's build should run BibTeX automatically. All required packages are standard TeX Live packages; there are no custom fonts, external scripts, shell-escape requirements, or image-generation dependencies. The bibliography uses `natbib` and `unsrtnat`, not Biber.

**Local build status:** no `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `bibtex`, or `tectonic` executable was found on PATH; common local TeX installation directories were also absent. Compilation was therefore not attempted in this environment. A [compiled report PDF](../mnema.pdf) is supplied, but the static source checks documented here do not replace a clean local compilation. After compiling, inspect `main.log` for undefined references/citations and overfull boxes, and verify figure placement and the final page count.

## Empirical authority and source audit

The audit was completed before drafting. The sole authority for reported experimental results is [the full BENCHMARK_REPORT.md](../../results/research_benchmark/full/BENCHMARK_REPORT.md). Supporting artifacts were inspected to verify transcription, interpretation, and provenance. Historical results and the quick benchmark were not used as full-data evidence. No new numerical experiment or new result plot was created.

| Report content | Repository sources inspected | What they establish |
| --- | --- | --- |
| Project scope and evidence boundaries | [root README](../../README.md), [architecture](../ARCHITECTURE.md), [research protocol](../RESEARCH_BENCHMARK.md), [getting started](../GETTING_STARTED.md) | Current prototype, documented caveats, protocol, and separate research/package environments |
| Current architecture | [model.py](../../mnema/model.py), [encoder.py](../../mnema/encoder.py), [separator.py](../../mnema/separator.py), [store.py](../../mnema/store.py) | TTFS with 16 bins; 16,384 separator units; top-64 indices; timing collapse; associative class votes and budget enforcement |
| Adaptive learning and consolidation | [cortex.py](../../mnema/cortex.py), [readout.py](../../mnema/readout.py), [modulators.py](../../mnema/modulators.py), [consolidate.py](../../mnema/consolidate.py) | Shallow projection; four synaptic planes; membrane readout; logistic arbitration; controller gates; single-row consolidation |
| Protocol and exact settings | [full config.json](../../results/research_benchmark/full/config.json), [runner](../../experiments/run_research_benchmark.py), [dataset/stream implementation](../../benchmarks/research_split_mnist.py) | 60,000/10,000 images; fixed five-task order; seeds 0–9; all ten outputs; dimensions, hyperparameters, versions, hashes |
| Baseline mechanics | [research replay](../../baselines/research_replay.py), [EWC](../../baselines/ewc.py), [DER++](../../baselines/derpp.py), full report and protocol documentation | Reservoir policies, strict replay-buffer accounting, Fisher approximation/extra reads, pre-update logits, independent DER++ replay draws |
| Evaluation and metrics | [research_state.py](../../experiments/research_state.py), [research_results.py](../../experiments/research_results.py) | Per-image checkpoint restoration, preserved membrane state, RNG restoration, hash/reverse-order checks, macro accuracy, exact forgetting maximum, sample SD |
| Results and tables | [full report](../../results/research_benchmark/full/BENCHMARK_REPORT.md), [summary JSON](../../results/research_benchmark/full/summaries/summary.json), [CSV](../../results/research_benchmark/full/summaries/summary.csv), [summary table](../../results/research_benchmark/full/summaries/summary_table.md) | All six accuracy/forgetting means and SDs, memory categories, projected inference values, FastStore occupancy and exceedance counts |
| Memory and efficiency | [research_memory.py](../../experiments/research_memory.py), [counters.py](../../instrument/counters.py), [ASIC 45 nm card](../../instrument/tech/asic_45nm.yaml), full report | Unique array payload and exclusions; 25 versus 28 bytes per row; incomplete operation-counter-based projection |
| Figures and captions | [original plots](../../results/research_benchmark/full/plots/), [plotting code](../../experiments/plot_research_benchmark.py) | Actual plotted quantities, error-bar semantics, original scientific figures |
| Provenance and execution | [full bundle README](../../results/research_benchmark/full/README.md), [import validation](../../results/research_benchmark/full/IMPORT_VALIDATION.json), [import manifest](../../results/research_benchmark/full/IMPORT_MANIFEST.json), [runbook](../../experiments/RESEARCH_BENCHMARK.md), [research pins](../../experiments/research_requirements.txt) | Saved-artifact audit without rerunning models, source mismatch, canonical command, preservation requirements, recorded environment |

Some prose in the repository describes `accuracy_vs_memory` as a resident-memory plot. Its actual horizontal axis and plotting code use **auxiliary content**. The report follows the actual figure and code, explicitly corrects that interpretation in its caption, and gives resident memory separately in the tables. The retained heatmap footer mentions error bars generically; its caption clarifies that the heatmaps show means only.

Implementation comments were checked against executable behavior. In particular, the stored cortex `hidden_dim` does not create a hidden layer, the controller's `sigma` and readout's `a2` are unused, and consolidation does not preserve complete historical sparse codes. The selected energy card also lacks coefficients for the recorded synaptic-operation and neuron-update counters.

## Figures and tables

| Number | Figure | Origin |
| --- | --- | --- |
| 1 | Current data flow and training control | TikZ in `main.tex`, derived from current architecture docs and code; no empirical values generated |
| 2 | Mean retention matrices | Byte-for-byte copy of `results/research_benchmark/full/plots/retention_matrices.pdf` |
| 3 | Final accuracy versus auxiliary payload | Byte-for-byte copy of `results/research_benchmark/full/plots/accuracy_vs_memory.pdf`; **not total resident memory** |

| Number | Table | Source |
| --- | --- | --- |
| 1 | Six-method accuracy, forgetting, resident arrays, and projected native inference | Full report, Section 4 |
| 2 | Model/adaptive, fixed, auxiliary content, and allocated memory decomposition | Full report, Section 4 |
| 3 | Configured FastStore budget, row accounting, occupied payload, allocated arrays, total state | Full report, Section 7 |

Copied PDF SHA-256 hashes (each equals the source hash):

```text
retention_matrices.pdf
04786c1f4362f5adede9a8800334d5224f5356cc8f834b7c9d6dada148a128b7

accuracy_vs_memory.pdf
60395bb4b2901b2348e847f7f420128c2025590d6d65f0fdcfc681d9fd37f919
```

## Reproducibility boundary

Repository: [ujandey/mnema](https://github.com/ujandey/mnema).

- Recorded benchmark source: `fe86ec1c6abc600dda8ec50565a551af4e5434bd`.
- Checkout audited for this report: `0d317c0b75f82fe9a599a84a469684e991259806`.
- Configuration ID: `ab869e57feea62e93198f72fedde18182dc93a28a72d4875e966067d676027a8`.
- Configuration created: `2026-09-11T19:38:47.232647+00:00`.
- Recorded software: Python 3.11.7, NumPy 2.2.6, Matplotlib 3.10.3, PyYAML 6.0.2; one BLAS thread.

The verified full command, run from the repository root, is:

```sh
python experiments/run_research_benchmark.py --mode full --seeds 10
```

This command is documented for reproduction; it was **not executed** when preparing the report. It targets a directory that already contains the saved evidence. The current source differs from the benchmark manifest, so normal validation/resume rejects that bundle here. Restore the recorded source bytes and environment for exact-source reproduction; use a separate checkout/output context for a new current-source experiment. Do not modify source hashes or overwrite benchmark data to make validation pass. The repository has since standardized its package metadata on Python 3.11.7; the recorded benchmark environment remains unchanged.

The repository contains raw per-pair results, combined records, per-seed train/test indices, source and dataset checksums, configuration, summaries, worker records, and saved validation. The import audit's successful saved-artifact checks are not an independent rerun of model inference. That audit predates subsequent license-header changes and should not be represented as a current byte-identical source check.

## References and intentionally omitted claims

Original academic references were checked against the [McCloskey–Cohen publisher record](https://www.sciencedirect.com/science/article/pii/S0079742108605368), [original EWC paper](https://doi.org/10.1073/pnas.1611835114), [DER/DER++ proceedings record](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), and [Benna–Fusi publisher record](https://doi.org/10.1038/nn.4401). They supply background/attribution only, not external performance results. A complementary-learning-systems claim is not developed, so no such reference is added merely to enlarge the bibliography.

The report intentionally omits unsupported claims of:

- statistical significance, state-of-the-art performance, tuned-best-method status, or solving catastrophic forgetting;
- a total 64 KiB MNEMA model, a correctly enforced 64 KiB active payload, exact memory matching, packed deployable sparse storage, or process/peak RAM inferred from array payload;
- measured power, measured device energy, a physical energy-efficiency advantage, neuromorphic deployment, or a cross-method end-to-end training-energy ranking;
- preservation of TTFS timing downstream, a multilayer cortex, functioning surprise-based adaptation, age-weighted consolidation sampling, or replay of complete historical 64-index codes;
- causal attribution of retention to any one component, theoretical memory-scaling laws, category-level one-shot learning, privacy/non-invertibility guarantees, or historical 112-byte traces;
- an independent benchmark rerun, exact current-checkout reproduction, or guaranteed cross-platform floating-point equality;
- completed Split-FashionMNIST, multiple-order experiments, architecture ablations, or hardware studies. These are explicitly labeled proposed future work, not established project results or delivery commitments.

## Validation performed

Read-only checks established that:

- All six final-accuracy and forgetting mean/SD pairs match the current full report and summary JSON at displayed precision.
- All main-table resident-memory and inference-projection means match the saved summary; every memory-decomposition table cell matches its corresponding field.
- Full-data sizes total 60,000 training / 10,000 test images, with seeds 0–9 and 60 completed seed–method pairs.
- The two copied PDF figures match their source bytes and SHA-256 hashes and are readable single-page PDF files.
- All **152 original benchmark files** match the sizes and SHA-256 hashes in the preserved import manifest.
- LaTeX/BibTeX braces and LaTeX environments are balanced; citation keys, labels, cross-references, and included figure paths resolve statically. All six bibliography entries are cited.
- The working-tree change is confined to `docs/technical_report/`.

These are document and artifact consistency checks. They do not replace LaTeX compilation, visual inspection of the final PDF, or experimental reproduction. No research-code tests or benchmark runs were needed for this documentation-only change.

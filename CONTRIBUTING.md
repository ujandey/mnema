# Contributor guide

[Project overview](README.md) · [Documentation index](docs/README.md)

Contributions are welcome under the [Apache License, Version 2.0](LICENSE). Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in MNEMA are provided under that license.

## Development setup

Use the [Python 3.11.7 research environment](docs/GETTING_STARTED.md) for protocol work. The package metadata and pinned research requirements use the same dependency versions.

```bash
python -m pip install -r experiments/research_requirements.txt
python -m unittest discover -s tests -p "test_research*.py" -v
```

The tests use small synthetic fixtures and temporary artifacts. The lightweight CI workflow runs the research tests and calibration check on every push and pull request; full benchmark workflows remain manual.

## Scope a change

Identify whether the change affects model behavior, experimental protocol, execution infrastructure, generated reports, or documentation. Keep result families distinguishable in the final implementation and description. A model or protocol change should create newly labeled evidence rather than silently replacing the meaning of existing results.

The research source manifest hashes model, baseline, loader, instrument, technology-card, and research-harness files. Editing even a report generator can make old artifacts fail source validation. Markdown-only changes are outside that manifest. See [source_manifest](experiments/run_research_benchmark.py) before deciding how to preserve comparability.

## Preserve research invariants

When changing benchmark code, verify the invariants relevant to the change:

- Methods share the same per-seed external stream and dense baseline initialization.
- Predictions use all ten classes without test labels or task identifiers as inputs.
- Held-out evaluation restores model/RNG state and retains reverse-order checks.
- Forgetting excludes pre-learning and final checkpoints from its maximum, and permits negative values.
- Memory distinguishes active content, allocated capacity, adaptive state, and fixed scaffold.
- Counter projections remain labeled as projected and retain their coverage limits.
- Atomic persistence and digest checks reject incomplete, duplicate, or mismatched artifacts.

Task boundaries used by EWC during training and the additional Fisher pass are part of the declared protocol. Changes to those semantics require updated documentation and evidence.

## Validate proportionately

| Change | Useful checks |
| --- | --- |
| Documentation | Check relative links, anchors, CLI flags, snippets, units, and source attribution |
| Gradient, replay, or metric behavior | Research unit tests and focused tests for the changed behavior |
| Inference state | State hash / reverse-order tests in the research suite |
| Distributed collection | [test_research_actions.py](tests/test_research_actions.py) with tiny fixture shards |
| Instrument changes | Unit tests plus `python experiments/e0_calibrate.py` and the applicable arithmetic diagnostic |
| End-to-end integration | Explicit quick run in a fresh or deliberately archived mode directory; validate that bundle |

Run the full protocol only when the change requires new full-data evidence and you have the necessary compute resources. Do not use a 60-pair experiment as a routine documentation check. Do not run historical exporters or benchmarks incidentally; they overwrite tracked outputs.

## Maintain documentation

- Put onboarding in [GETTING_STARTED.md](docs/GETTING_STARTED.md), scientific definitions in [RESEARCH_BENCHMARK.md](docs/RESEARCH_BENCHMARK.md), and command operations in the [runbook](experiments/RESEARCH_BENCHMARK.md).
- Keep the root README readable as an overview and route detail through the [documentation index](docs/README.md).
- Use repository-relative links, UTF-8 Markdown, language-tagged code fences, and explicit output paths.
- Give a command's prerequisites, side effects, and success condition. Distinguish seed indices from seed counts, full data from debug subsets, and validation from reproduction.
- Link numerical claims to a result bundle and retain mode, sample counts, seed count, units, uncertainty, and accounting boundaries.
- Use the [completed full bundle](results/research_benchmark/full/README.md) as the primary Split-MNIST evidence. Preserve imported files and their checksums; place current-source audit notes alongside the original record. Keep quick and historical numbers labeled by their own protocols.
- Keep the `RESEARCH_BENCHMARK_START` / `RESEARCH_BENCHMARK_END` markers intact. The report generator replaces only that block when explicitly requested.
- Treat generated Markdown/CSV as derived views. Curated context belongs in the result guide and protocol; generators can overwrite editorial notes in snapshots.

Do not invent contact details, performance guarantees, hardware support, or stable APIs. Preserve copyright attributions and keep license notices consistent with the root Apache 2.0 license.

## Prepare a reviewable change

```bash
git diff --check
git diff --stat
git status --short
```

In the change description, state the concrete behavior or reader problem addressed, relevant validation, and any remaining limitation. For new results, identify the exact source/configuration and whether the evidence is quick or full. Include complete artifact provenance rather than a favorable table alone.

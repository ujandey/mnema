# MNEMA documentation

[Project overview](../README.md)

This documentation describes the code and artifacts in this checkout. Start with the task you need to complete; experiment settings and saved JSON take precedence over narrative summaries when reproducing a run.

## Learn and install

| Guide | What you will learn |
| --- | --- |
| [Getting started](GETTING_STARTED.md) | Choose the correct Python environment, install dependencies, verify setup, and run a debug experiment |
| [Architecture](ARCHITECTURE.md) | Follow one sample through the model, understand learning and inference state, and inspect implementation limits |
| [Python API](API.md) | Construct a model, train one example, inspect predictions, and use counters correctly |
| [Glossary](GLOSSARY.md) | Decode Class-IL, retention matrices, sparse operations, and memory terms |

## Run and interpret

| Guide | What it covers |
| --- | --- |
| [Research protocol and results](RESEARCH_BENCHMARK.md) | Dataset, methods, metrics, evaluation isolation, cloud execution, and the separate full-data record |
| [Benchmark runbook](../experiments/RESEARCH_BENCHMARK.md) | Exact commands, CLI defaults, resume behavior, and regeneration |
| [Experiment catalog](../experiments/README.md) | Diagnostic scripts, outputs, and historical versus research scope |
| [Result artifact guide](../results/README.md) | Provenance, artifact layout, generated files, and accounting caveats |

## Maintain and troubleshoot

- [Contributor guide](../CONTRIBUTING.md): development checks, scientific invariants, and documentation maintenance.
- [Troubleshooting](TROUBLESHOOTING.md): environment mismatches, dataset errors, resume refusals, and missing artifacts.
- [License](../LICENSE): repository access and use terms.

## Sources of truth

| Question | Source |
| --- | --- |
| What does the model execute? | [mnema/model.py](../mnema/model.py) and the component modules it calls |
| Which dependencies reproduce the research environment? | [research_requirements.txt](../experiments/research_requirements.txt) and the run's `config.json` |
| What is the research protocol? | [Runner](../experiments/run_research_benchmark.py), [task loader](../benchmarks/research_split_mnist.py), and saved configuration |
| How are metrics and evaluation implemented? | [research_results.py](../experiments/research_results.py) and [research_state.py](../experiments/research_state.py) |
| What was observed in a particular run? | That run's raw JSON, predictions, indices, summary, and validation record |
| What can be inferred from energy or memory numbers? | [Accounting boundaries](RESEARCH_BENCHMARK.md#memory-and-energy-accounting) |

Generated reports are snapshots. Regenerating them does not rerun training; rerunning an experiment can replace saved outputs. A `full_results.json` filename alone does not establish that a run used full data: inspect `config.mode` and seed coverage.

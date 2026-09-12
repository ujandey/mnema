## Research Benchmark

<!-- Generated root-README fragment. Links below resolve from the repository root,
     not from this summaries directory. Open ../BENCHMARK_REPORT.md for a standalone
     report. Regeneration replaces this file; see results/README.md for context. -->

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

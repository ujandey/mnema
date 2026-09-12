## Research Benchmark

**Full-data research protocol.**

Full Split-MNIST, five sequential digit pairs, one external training pass, ten-class inference without task IDs; default full seeds 0–9. Methods: Naive MLP, reservoir Replay-300 (Research), Replay-64KiB, EWC, DER++-300 and frozen MNEMA.

Run on a stronger machine: `python experiments/run_research_benchmark.py --mode full --seeds 10`. Resume with the same command; add `--force` to archive and rerun. Development: `python experiments/run_research_benchmark.py --quick`.

| Method | Final ACC (%) | Forgetting (pp) | Model/adaptive (B) | Fixed scaffold (B) | Auxiliary content (B) | Auxiliary allocated (B) | Total resident arrays (B) | Native projected inference (µJ/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive MLP | 19.79 ± 0.06 | 99.52 ± 0.12 | 814120 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 814120 ± 0 | 1.768 ± 0.000 |
| Replay-300 (Research) | 82.44 ± 1.33 | 20.58 ± 1.62 | 814120 ± 0 | 0 ± 0 | 235516 ± 0 | 235516 ± 0 | 1049636 ± 0 | 1.768 ± 0.000 |
| Replay-64KiB | 67.59 ± 2.07 | 39.30 ± 2.60 | 814120 ± 0 | 0 ± 0 | 65171 ± 0 | 65171 ± 0 | 879291 ± 0 | 1.768 ± 0.000 |
| EWC | 19.98 ± 0.36 | 99.12 ± 0.40 | 814120 ± 0 | 0 ± 0 | 6512960 ± 0 | 6512960 ± 0 | 7327080 ± 0 | 1.768 ± 0.000 |
| DER++-300 | 89.39 ± 0.60 | 12.12 ± 0.73 | 814120 ± 0 | 0 ± 0 | 247516 ± 0 | 247516 ± 0 | 1061636 ± 0 | 1.768 ± 0.000 |
| MNEMA | 77.12 ± 0.87 | 6.53 ± 1.41 | 3408032 ± 0 | 524316 ± 0 | 73388 ± 0 | 458752 ± 0 | 4391100 ± 0 | 0.316 ± 0.000 |

Memory columns are array payload; total = adaptive + fixed + allocated auxiliary. Native inference energy is an incomplete projection, not measured energy. MNEMA's configured 64 KiB limit is not an actual payload ceiling (25 bytes counted versus 28 bytes per active row). Stateless checkpoint evaluation is enforced per image. These results are not comparable directly with historical small-benchmark numbers.

[Detailed report](results/research_benchmark/full/BENCHMARK_REPORT.md) · [Generated summary](results/research_benchmark/full/summaries/summary.json)

![Accuracy](results/research_benchmark/full/plots/final_accuracy.png)
![Memory trade-off](results/research_benchmark/full/plots/accuracy_vs_memory.png)
![Retention](results/research_benchmark/full/plots/retention_matrices.png)

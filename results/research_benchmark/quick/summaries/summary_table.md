# Debug benchmark summary

[Detailed report](../BENCHMARK_REPORT.md) · [Summary JSON](summary.json) · [Artifact guide](../../../README.md)

This generated table describes the saved quick run only. Numerical values are unchanged; regeneration can overwrite this editorial context. Memory is array payload and energy is a partial native projection. Auxiliary content and allocated auxiliary memory are alternative views: total resident arrays = adaptive + fixed + allocated auxiliary.

**DEBUG RESULTS ONLY — reduced data, not research evidence. The full ten-seed experiment has NOT been completed by this quick run.**

† One seed: standard deviation is undefined; values are single-seed observations.

| Method | Final ACC (%) | Forgetting (pp) | Model/adaptive (B) | Fixed scaffold (B) | Auxiliary content (B) | Auxiliary allocated (B) | Total resident arrays (B) | Native projected inference (µJ/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive MLP | 19.40† | 98.25† | 814120† | 0† | 0† | 0† | 814120† | 1.768† |
| Replay-300 (Research) | 65.40† | 37.50† | 814120† | 0† | 235516† | 235516† | 1049636† | 1.768† |
| Replay-64KiB | 67.00† | 35.00† | 814120† | 0† | 65171† | 65171† | 879291† | 1.768† |
| EWC | 19.60† | 98.00† | 814120† | 0† | 6512960† | 6512960† | 7327080† | 1.768† |
| DER++-300 | 72.00† | 26.75† | 814120† | 0† | 247516† | 247516† | 1061636† | 1.768† |
| MNEMA | 71.20† | 24.25† | 3408032† | 524316† | 73388† | 458752† | 4391100† | 0.316† |

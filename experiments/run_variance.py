# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from benchmarks.split_mnist import load_split_mnist
from baselines.mlp import BaselineMLP
from mnema.model import MNEMA
from instrument.counters import EnergyInstrument
from experiments.run_split_mnist_benchmark import evaluate_retention


def run_seed(tasks, seed):
    """Runs all three models under a fixed seed and returns their metrics."""
    out = {}
    specs = {
        "MLP (Naive)": lambda: (BaselineMLP(use_replay=False), False),
        "MLP (+Replay Buffer)": lambda: (BaselineMLP(use_replay=True, buffer_size=300), False),
        "MNEMA (Full Engine)": lambda: (MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536), True),
    }

    for name, build in specs.items():
        np.random.seed(seed)
        model, is_mnema = build()
        instrument = EnergyInstrument(tech_card_path="instrument/tech/asic_45nm.yaml")
        R = np.zeros((5, 5), dtype=np.float32)

        with instrument:
            for task_id in range(5):
                train_x = tasks[task_id]["train_x"][:80]
                train_y = tasks[task_id]["train_y"][:80]
                for x, y in zip(train_x, train_y):
                    if is_mnema:
                        model.step(x, y=int(y), is_training=True, instrument=instrument)
                    else:
                        model.train_step(x, y=int(y), instrument=instrument)
                for j, acc in enumerate(evaluate_retention(model, tasks, task_id, is_mnema, instrument)):
                    R[task_id, j] = acc

        out[name] = {
            "final_acc": float(np.mean(R[4, :]) * 100),
            "forgetting": float(np.mean([np.max(R[:, j]) - R[4, j] for j in range(4)]) * 100),
            "energy_uj": instrument.project_energy()["energy_microjoules"],
            "macs": instrument.counts.macs,
            "synops": instrument.counts.synops,
            "adds": instrument.counts.adds,
        }
    return out


def main(n_seeds=10):
    print(f"=== MNEMA Variance Study: {n_seeds} seeds x 3 models (Split-MNIST Class-IL) ===\n")
    tasks = load_split_mnist()
    runs = []
    for seed in range(n_seeds):
        print(f"  seed {seed} ...", flush=True)
        runs.append(run_seed(tasks, seed))

    summary = {}
    for name in runs[0]:
        summary[name] = {}
        for metric in ("final_acc", "forgetting", "energy_uj", "macs", "synops", "adds"):
            vals = np.array([r[name][metric] for r in runs], dtype=np.float64)
            summary[name][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

    os.makedirs("results", exist_ok=True)
    with open("results/variance_results.json", "w") as f:
        json.dump({"n_seeds": n_seeds, "per_seed": runs, "summary": summary}, f, indent=2)

    print(f"\n{'Model':<24}{'ACC (mean+-std)':<22}{'FM (mean+-std)':<22}{'Energy uJ':<14}")
    print("-" * 82)
    for name, s in summary.items():
        acc = f"{s['final_acc']['mean']:.1f} +- {s['final_acc']['std']:.1f}%"
        fm = f"{s['forgetting']['mean']:.1f} +- {s['forgetting']['std']:.1f}%"
        en = f"{s['energy_uj']['mean']:.1f}"
        print(f"{name:<24}{acc:<22}{fm:<22}{en:<14}")

    print("\nSaved -> results/variance_results.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)

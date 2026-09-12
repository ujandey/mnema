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

def evaluate_retention(model, tasks, up_to_task, is_mnema=False, instrument=None):
    accs = []
    for t in range(up_to_task + 1):
        test_x = tasks[t]["test_x"][:50]
        test_y = tasks[t]["test_y"][:50]
        correct = 0
        for x, y in zip(test_x, test_y):
            if is_mnema:
                out = model.step(x, is_training=False, instrument=instrument)
                pred = out["prediction"]
            else:
                _, probs = model.forward(x.reshape(1, -1), instrument=instrument)
                pred = np.argmax(probs)
            if pred == y:
                correct += 1
        accs.append(correct / len(test_y))
    return accs

def run_benchmark(seed=0):
    print("=== COGNX MNEMA: Class-Incremental Split-MNIST Benchmark ===")
    print(f"(seed={seed}; run experiments/run_variance.py for multi-seed error bars)\n")
    tasks = load_split_mnist()

    # Models are constructed lazily under a fixed seed so each run is reproducible.
    builders = {
        "MLP (Naive)": (lambda: BaselineMLP(use_replay=False), False),
        "MLP (+Replay Buffer)": (lambda: BaselineMLP(use_replay=True, buffer_size=300), False),
        "MNEMA (Full Engine)": (lambda: MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536), True)
    }

    results = {}

    for name, (build, is_mnema) in builders.items():
        print(f"--- Benchmarking: {name} ---")
        np.random.seed(seed)
        model = build()
        instrument = EnergyInstrument(tech_card_path="instrument/tech/asic_45nm.yaml")
        
        # Retention matrix R[t, j]
        R = np.zeros((5, 5), dtype=np.float32)

        with instrument:
            for task_id in range(5):
                train_x = tasks[task_id]["train_x"][:80]  # Online streaming sample
                train_y = tasks[task_id]["train_y"][:80]

                # Online stream training
                for x, y in zip(train_x, train_y):
                    if is_mnema:
                        model.step(x, y=int(y), is_training=True, instrument=instrument)
                    else:
                        model.train_step(x, y=int(y), instrument=instrument)

                # Evaluate retention across all seen tasks
                task_accs = evaluate_retention(model, tasks, task_id, is_mnema=is_mnema, instrument=instrument)
                for j, acc in enumerate(task_accs):
                    R[task_id, j] = acc

        # Compute CL Metrics
        final_acc = float(np.mean(R[4, :]))
        forgetting = float(np.mean([np.max(R[:, j]) - R[4, j] for j in range(4)]))
        proj = instrument.project_energy()

        results[name] = {
            "final_acc": round(final_acc * 100, 2),
            "forgetting": round(forgetting * 100, 2),
            "macs": instrument.counts.macs,
            "synops": instrument.counts.synops,
            "adds": instrument.counts.adds,
            "energy_uj": round(proj["energy_microjoules"], 2),
            "retention_matrix": R.tolist()
        }

        print(f"Final Accuracy (ACC) : {results[name]['final_acc']}%")
        print(f"Forgetting (FM)      : {results[name]['forgetting']}%")
        print(f"Projected Energy     : {results[name]['energy_uj']} uJ\n")

    # Save results to disk
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Benchmark Finished. Results saved to results/benchmark_results.json ===")

if __name__ == "__main__":
    run_benchmark(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
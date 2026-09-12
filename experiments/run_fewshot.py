# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

"""
Few-shot adaptation probe: how many examples does each system need to acquire a
NEW class that appears after deployment, without disturbing what it already knows?

This is the operating regime the product actually targets (a device meeting an object
it has never seen), and it is not measured by the Split-MNIST retention benchmark.
"""
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from benchmarks.split_mnist import load_split_mnist
from baselines.mlp import BaselineMLP
from mnema.model import MNEMA

SHOTS = [1, 2, 5, 10, 20, 50]
N_EVAL = 100


def eval_model(model, is_mnema, xs, ys):
    correct = 0
    for x, y in zip(xs, ys):
        if is_mnema:
            pred = model.step(x, is_training=False)["prediction"]
        else:
            _, probs = model.forward(x.reshape(1, -1))
            pred = int(np.argmax(probs))
        correct += int(pred == y)
    return correct / len(ys)


def run(seed, tasks):
    """Pretrain on classes 0-7, then introduce classes 8/9 with k shots."""
    old_x = np.concatenate([tasks[t]["train_x"][:80] for t in range(4)])
    old_y = np.concatenate([tasks[t]["train_y"][:80] for t in range(4)])
    new_x, new_y = tasks[4]["train_x"], tasks[4]["train_y"]

    old_test_x = np.concatenate([tasks[t]["test_x"][:50] for t in range(4)])
    old_test_y = np.concatenate([tasks[t]["test_y"][:50] for t in range(4)])
    new_test_x, new_test_y = tasks[4]["test_x"][:N_EVAL], tasks[4]["test_y"][:N_EVAL]

    out = {}
    for label, is_mnema in (("MNEMA", True), ("MLP (Naive)", False), ("MLP (+Replay)", False)):
        row = {}
        for k in SHOTS:
            np.random.seed(seed)
            if is_mnema:
                model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
            else:
                model = BaselineMLP(use_replay=(label == "MLP (+Replay)"), buffer_size=300)

            for x, y in zip(old_x, old_y):
                if is_mnema:
                    model.step(x, y=int(y), is_training=True)
                else:
                    model.train_step(x, y=int(y))

            # Introduce the new class with exactly k examples per class.
            for cls in (8, 9):
                sel = new_x[new_y == cls][:k]
                for x in sel:
                    if is_mnema:
                        model.step(x, y=cls, is_training=True)
                    else:
                        model.train_step(x, y=cls)

            row[k] = {
                "new_class_acc": eval_model(model, is_mnema, new_test_x, new_test_y) * 100,
                "old_class_acc": eval_model(model, is_mnema, old_test_x, old_test_y) * 100,
            }
        out[label] = row
    return out


def main(n_seeds=3):
    print("=== Few-Shot New-Class Acquisition (pretrain classes 0-7, introduce 8/9) ===\n")
    tasks = load_split_mnist()
    runs = [run(s, tasks) for s in range(n_seeds)]

    summary = {}
    for label in runs[0]:
        summary[label] = {}
        for k in SHOTS:
            new = np.array([r[label][k]["new_class_acc"] for r in runs])
            old = np.array([r[label][k]["old_class_acc"] for r in runs])
            summary[label][k] = {
                "new_mean": float(new.mean()), "new_std": float(new.std(ddof=1)) if n_seeds > 1 else 0.0,
                "old_mean": float(old.mean()), "old_std": float(old.std(ddof=1)) if n_seeds > 1 else 0.0,
            }

    hdr = "shots/class".ljust(14) + "".join(f"{lbl:<26}" for lbl in summary)
    print(hdr)
    print("(new-class acc | retained old acc)".ljust(14) + "-" * (26 * len(summary)))
    for k in SHOTS:
        line = f"{k:<14}"
        for lbl in summary:
            s = summary[lbl][k]
            line += f"{s['new_mean']:>5.1f}% | {s['old_mean']:>5.1f}%".ljust(26)
        print(line)

    os.makedirs("results", exist_ok=True)
    with open("results/fewshot_results.json", "w") as f:
        json.dump({"n_seeds": n_seeds, "shots": SHOTS, "summary": summary}, f, indent=2)
    print("\nSaved -> results/fewshot_results.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

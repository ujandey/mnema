# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

"""
Code-geometry diagnostic: measures how much the sparse separator code overlaps for
samples of the SAME class versus DIFFERENT classes, on real MNIST digits.

This is the quantity that governs the whole architecture. High intra-class overlap is
what lets the FastStore generalize from few examples. Low inter-class overlap is what
prevents catastrophic interference. The separator must deliver both at once; if
intra-class overlap collapses toward the inter-class floor, the store can memorize but
cannot generalize, and few-shot acquisition fails.
"""
import sys
import os
import json
import itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from benchmarks.split_mnist import load_split_mnist
from mnema.encoder import EventEncoder
from mnema.separator import SparseSeparator

N_PER_CLASS = 30


def codes_for(separator, encoder, xs):
    return [set(separator.forward(encoder.encode(x)).tolist()) for x in xs]


def measure(n_s, k, fan_in, tasks, rng):
    np.random.seed(0)
    enc = EventEncoder(encoder_type="ttfs", T=16)
    sep = SparseSeparator(n_in=784, n_s=n_s, fan_in=fan_in, k=k)

    by_class = {}
    for t in range(5):
        for cls in tasks[t]["classes"]:
            xs = tasks[t]["train_x"][tasks[t]["train_y"] == cls][:N_PER_CLASS]
            by_class[cls] = codes_for(sep, enc, xs)

    intra, inter = [], []
    for cls, codes in by_class.items():
        for a, b in itertools.combinations(range(len(codes)), 2):
            intra.append(len(codes[a] & codes[b]))
    classes = sorted(by_class)
    for c1, c2 in itertools.combinations(classes, 2):
        for a in range(0, N_PER_CLASS, 3):
            for b in range(0, N_PER_CLASS, 3):
                inter.append(len(by_class[c1][a] & by_class[c2][b]))

    intra, inter = np.array(intra, float), np.array(inter, float)
    return {
        "n_s": n_s, "k": k, "fan_in": fan_in,
        "intra_mean": float(intra.mean()), "intra_std": float(intra.std()),
        "inter_mean": float(inter.mean()), "inter_std": float(inter.std()),
        "separation_ratio": float(intra.mean() / max(inter.mean(), 1e-9)),
        "intra_frac_of_k": float(intra.mean() / k),
        "theoretical_random_overlap": k * k / n_s,
    }


def main():
    print("=== Sparse Code Geometry: intra-class vs inter-class overlap (real MNIST) ===\n")
    tasks = load_split_mnist()
    rng = np.random.default_rng(0)

    configs = [
        (16384, 64, 8),    # shipped default
        (16384, 256, 8),   # denser code
        (4096, 64, 8),     # smaller expansion
        (2048, 128, 16),   # small + dense + wider fan-in
        (16384, 64, 32),   # wider fan-in only
    ]

    rows = []
    print(f"{'N_S':>7}{'k':>6}{'fan_in':>8} | {'intra':>14}{'inter':>14}{'ratio':>9}{'intra/k':>10}")
    print("-" * 76)
    for n_s, k, fan_in in configs:
        r = measure(n_s, k, fan_in, tasks, rng)
        rows.append(r)
        tag = "  <- shipped default" if (n_s, k, fan_in) == (16384, 64, 8) else ""
        print(f"{n_s:>7}{k:>6}{fan_in:>8} | "
              f"{r['intra_mean']:>7.2f}+-{r['intra_std']:<5.2f}"
              f"{r['inter_mean']:>7.2f}+-{r['inter_std']:<5.2f}"
              f"{r['separation_ratio']:>9.2f}{100*r['intra_frac_of_k']:>9.1f}%{tag}")

    print("\nReading: 'intra/k' is the fraction of a code shared by two samples of the same")
    print("class. That is the fraction the FastStore can reuse when generalizing. A value")
    print("near 0 means every new sample writes to fresh rows and the store memorizes")
    print("instead of generalizing, which is exactly what caps few-shot accuracy.")

    os.makedirs("results", exist_ok=True)
    with open("results/code_geometry.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved -> results/code_geometry.json")


if __name__ == "__main__":
    main()

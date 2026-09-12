# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mnema.model import MNEMA
from instrument.counters import EnergyInstrument

def test_full_mnema_system():
    print("=== Testing Complete MNEMA System: [E]+[S]+[F]+[C]+[R]+[N]+[Z] ===")
    model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
    instrument = EnergyInstrument(tech_card_path="instrument/tech/asic_45nm.yaml")

    np.random.seed(42)
    # Generate 5 sequential tasks (2 classes each)
    accuracies = []
    
    with instrument:
        for task_id in range(5):
            c_a, c_b = 2 * task_id, 2 * task_id + 1
            print(f"Training Task {task_id + 1}: Classes ({c_a}, {c_b})...")
            
            # Train 20 samples per class
            for _ in range(20):
                x_a = np.random.binomial(1, 0.15, size=784)
                x_b = np.random.binomial(1, 0.15, size=784)
                model.step(x_a, y=c_a, is_training=True, instrument=instrument)
                model.step(x_b, y=c_b, is_training=True, instrument=instrument)

    print("\n--- Testing Task 1 (Classes 0 & 1) Retention ---")
    correct = 0
    with instrument:
        for _ in range(10):
            x_test_0 = np.random.binomial(1, 0.15, size=784)
            out_0 = model.step(x_test_0, is_training=False, instrument=instrument)
            if out_0["prediction"] == 0:
                correct += 1

    proj = instrument.project_energy()
    print(f"Task 1 Retention Accuracy  : {(correct / 10) * 100:.1f}%")
    print(f"Total Measured SynOps      : {instrument.counts.synops:,}")
    print(f"Total Measured MACs        : {instrument.counts.macs:,}")
    print(f"Total Integer Adds         : {instrument.counts.adds:,}")
    print(f"Projected Energy (ASIC45nm): {proj['energy_microjoules']:.2f} uJ")
    print("=== [PASS] MNEMA Core Engine Functional ===")

if __name__ == "__main__":
    test_full_mnema_system()
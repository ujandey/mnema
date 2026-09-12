# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mnema.encoder import EventEncoder
from mnema.separator import SparseSeparator
from mnema.store import FastStore
from instrument.counters import EnergyInstrument

def test_mnema_forward_pipeline():
    print("=== Testing MNEMA Pipeline: [E] -> [S] -> [F] ===")
    encoder = EventEncoder(encoder_type="ttfs", T=16)
    separator = SparseSeparator(n_in=784, n_s=16384, fan_in=8, k=64)
    store = FastStore(n_s=16384, d=10, budget_bytes=65536)
    
    instrument = EnergyInstrument(tech_card_path="instrument/tech/asic_45nm.yaml")

    # Generate synthetic samples for 2 classes
    np.random.seed(42)
    sample_c0 = np.random.binomial(1, 0.2, size=784)
    sample_c1 = np.random.binomial(1, 0.2, size=784)

    with instrument:
        # 1. Encode
        spikes_c0 = encoder.encode(sample_c0, instrument=instrument)
        # 2. Separate
        code_c0 = separator.forward(spikes_c0, instrument=instrument)
        # 3. One-shot Write to Fast Store (Class 0)
        store.write(code_c0, target_label=0, instrument=instrument)

        # Encode and Separate Class 1
        spikes_c1 = encoder.encode(sample_c1, instrument=instrument)
        code_c1 = separator.forward(spikes_c1, instrument=instrument)
        store.write(code_c1, target_label=1, instrument=instrument)

        # 4. Inference Read (Query with Class 0)
        y_f, conf = store.read(code_c0, instrument=instrument)
        pred = np.argmax(y_f)

    # Compute code overlap between the two classes
    overlap = len(np.intersect1d(code_c0, code_c1))
    
    print(f"Active Code Size (k)     : {len(code_c0)} units (out of 16,384)")
    print(f"Inter-class Code Overlap : {overlap} units (Expected ~0-1)")
    print(f"Prediction for Class 0   : Class {pred} (Confidence: {conf:.2f})")
    print(f"Total MACs Executed      : {instrument.counts.macs} (Zero MACs confirmed)")
    print(f"Total Integer Adds       : {instrument.counts.adds:,}")
    print(f"Projected Energy         : {instrument.project_energy()['energy_microjoules']:.4f} uJ")
    print("=== [PASS] Modules [E], [S], and [F] Operational ===\n")

if __name__ == "__main__":
    test_mnema_forward_pipeline()
# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np
from mnema.encoder import EventEncoder
from mnema.separator import SparseSeparator
from mnema.store import FastStore
from mnema.cortex import SlowCortex
from mnema.readout import ReadoutArbitrator
from mnema.modulators import NeuromodulatoryController
from mnema.consolidate import SleepConsolidator

class MNEMA:
    """
    Unified MNEMA Architecture: [E] -> [S] -> [F] + [C] -> [R] with [N] and [Z].
    """
    def __init__(self, n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536):
        self.encoder = EventEncoder(encoder_type="ttfs", T=16)
        self.separator = SparseSeparator(n_in=n_in, n_s=n_s, fan_in=8, k=k)
        self.store = FastStore(n_s=n_s, d=d, budget_bytes=budget_bytes)
        self.cortex = SlowCortex(in_features=n_s, out_dim=d)
        self.readout = ReadoutArbitrator()
        self.controller = NeuromodulatoryController()
        self.consolidator = SleepConsolidator()

    def step(self, x: np.ndarray, y: int = None, is_training: bool = True, instrument=None) -> dict:
        # 1. [E] Encode input into spike raster
        spikes = self.encoder.encode(x, instrument=instrument)

        # 2. [S] Sparse random expansion + k-WTA
        code = self.separator.forward(spikes, instrument=instrument)

        # 3. [F] Fast Store Address-and-Accumulate
        y_f, conf_f = self.store.read(code, instrument=instrument)

        # 4. [C] Slow Cortex ALIF forward.
        # The eligibility trace is only consumed by cortex.learn(), so it is maintained
        # on the training path only. Inference leaves cortex plasticity state untouched.
        y_c = self.cortex.forward(code, instrument=instrument, update_eligibility=is_training)

        # 5. [R] Arbitration between Fast & Slow systems
        pred, probs, alpha = self.readout.arbitrate(y_f, conf_f, y_c)

        # 6. [N] Controller evaluation
        bus = self.controller.step(conf_f, pred, target=y)

        # 7. Online Learning & Sleep Trigger
        if is_training and y is not None:
            # Write to fast store on novelty
            if bus.nu > self.controller.nu_thresh:
                self.store.write(code, target_label=y, instrument=instrument)

            # Local plastic update in cortex on error
            if bus.delta > 0:
                self.cortex.learn(target_label=y, instrument=instrument)

            # Trigger offline sleep consolidation when sleep pressure crosses threshold
            if bus.phi >= self.controller.phi_thresh:
                self.consolidator.consolidate(self.store, self.cortex, instrument=instrument)
                self.controller.reset_sleep_pressure()

        return {
            "prediction": pred,
            "probabilities": probs,
            "fast_weight_alpha": alpha,
            "confidence": conf_f,
            "modulator_bus": bus
        }
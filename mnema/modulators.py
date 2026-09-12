# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

from dataclasses import dataclass
import numpy as np

@dataclass
class ModulatorBus:
    nu: float = 0.0       # Novelty (Acetylcholine) -> triggers Fast Store writes
    delta: float = 0.0    # Error (Dopamine) -> modulates cortex weight updates
    sigma: float = 0.0    # Surprise (Norepinephrine) -> modulates learning rate
    phi: float = 0.0      # Sleep Pressure (Adenosine) -> triggers consolidation

class NeuromodulatoryController:
    def __init__(self, nu_thresh=0.15, phi_thresh=50.0):
        self.nu_thresh = nu_thresh
        self.phi_thresh = phi_thresh
        self.bus = ModulatorBus()

    def step(self, conf_f: float, pred: int, target: int = None) -> ModulatorBus:
        # Novelty: inverse of Fast Store retrieval confidence
        self.bus.nu = max(0.0, 1.0 - conf_f)

        # Supervised error signal
        if target is not None:
            self.bus.delta = 1.0 if pred != target else 0.0
        else:
            self.bus.delta = 0.0

        # Accumulate sleep pressure based on operational drive
        self.bus.phi += (self.bus.nu + self.bus.delta)

        return self.bus

    def reset_sleep_pressure(self):
        self.bus.phi = 0.0
# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np

class FastStore:
    def __init__(self, n_s=16384, d=10, eta_f=1, s_max=255, tau_f=5000, theta_novel=0.15, budget_bytes=65536):
        self.n_s = n_s
        self.d = d
        self.eta_f = eta_f
        self.s_max = s_max
        self.tau_f = tau_f
        self.theta_novel = theta_novel
        self.budget_bytes = budget_bytes

        # Fast memory states
        self.M = np.zeros((n_s, d), dtype=np.int16)
        self.s = np.zeros(n_s, dtype=np.float32)
        self.tau = np.zeros(n_s, dtype=np.uint32)
        self.t_clock = 0

    def read(self, active_indices: np.ndarray, instrument=None):
        """
        Pure address-and-accumulate over k active rows.
        """
        # Read M for k rows and sum
        y_f = self.M[active_indices].sum(axis=0).astype(np.float32)
        
        # Calculate confidence metric
        sorted_vals = np.sort(y_f)
        max1 = sorted_vals[-1]
        max2 = sorted_vals[-2] if len(sorted_vals) > 1 else 0.0
        conf = (max1 - max2) / (np.abs(max1) + 1e-5)

        if instrument is not None:
            # k * D integer additions, ZERO MACs
            instrument.adds += len(active_indices) * self.d
            instrument.sram_read_b += len(active_indices) * self.d * 2

        return y_f, conf

    def write(self, active_indices: np.ndarray, target_label: int, instrument=None):
        self.t_clock += 1
        
        # One-hot target vector
        v = np.zeros(self.d, dtype=np.int16)
        v[target_label] = 1

        for idx in active_indices:
            self.M[idx] += self.eta_f * v
            self.s[idx] = min(self.s_max, self.s[idx] + 1.0)
            self.tau[idx] = self.t_clock

        # Apply eviction if over hard budget
        self._enforce_budget()

        if instrument is not None:
            instrument.adds += len(active_indices) * self.d
            instrument.sram_write_b += len(active_indices) * (self.d * 2 + 5)
            instrument.weight_updates += len(active_indices)

    def _enforce_budget(self):
        # Occupancy: number of active memory rows * bytes per row
        active_rows = np.where(self.s > 0)[0]
        bytes_per_row = self.d * 2 + 5  # int16 weights + salience + timestamp
        total_occupancy = len(active_rows) * bytes_per_row

        if total_occupancy > self.budget_bytes:
            excess_rows = (total_occupancy - self.budget_bytes) // bytes_per_row + 1
            # Evict rows with lowest salience
            lowest_salience_indices = active_rows[np.argsort(self.s[active_rows])[:excess_rows]]
            self.M[lowest_salience_indices] = 0
            self.s[lowest_salience_indices] = 0
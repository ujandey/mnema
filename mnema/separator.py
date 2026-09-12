# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np

class SparseSeparator:
    def __init__(self, n_in=784, n_s=16384, fan_in=8, k=64, reserve_frac=0.10, eta_homeo=1e-3, f_target=0.004):
        self.n_in = n_in
        self.n_s = n_s
        self.fan_in = fan_in
        self.k = k
        self.reserve_frac = reserve_frac
        self.eta_homeo = eta_homeo
        self.f_target = f_target

        self.n_active_pool = int(n_s * (1.0 - reserve_frac))
        
        # Frozen sparse binary connectivity: W_S has exactly `fan_in` indices per row
        self.indices = np.zeros((self.n_s, self.fan_in), dtype=np.int32)
        for i in range(self.n_s):
            self.indices[i] = np.random.choice(self.n_in, size=self.fan_in, replace=False)

        # Homeostatic excitability threshold per unit
        self.theta = np.zeros(self.n_s, dtype=np.float32)
        self.theta[self.n_active_pool:] = 1e6  # Reserve pool locked
        
        # Running firing rate tracker
        self.running_rate = np.zeros(self.n_s, dtype=np.float32)

    def forward(self, input_spikes: np.ndarray, instrument=None) -> np.ndarray:
        """
        input_spikes: [T, N_in] or [N_in]
        Returns top-k active binary indices.
        """
        if input_spikes.ndim > 1:
            spike_vec = (np.sum(input_spikes, axis=0) > 0).astype(np.float32)
        else:
            spike_vec = input_spikes.astype(np.float32)

        # Fully Vectorized Fan-in Accumulation (0 Python loops)
        drive_pool = np.sum(spike_vec[self.indices[:self.n_active_pool]], axis=1) - self.theta[:self.n_active_pool]

        # Top-k Winner-Take-All selection
        top_k_sub_indices = np.argpartition(drive_pool, -self.k)[-self.k:]
        top_k_indices = top_k_sub_indices[np.argsort(-drive_pool[top_k_sub_indices])]

        # Update local homeostasis
        f_instant = np.zeros(self.n_s, dtype=np.float32)
        f_instant[top_k_indices] = 1.0
        self.running_rate = 0.99 * self.running_rate + 0.01 * f_instant
        self.theta[:self.n_active_pool] += self.eta_homeo * (self.running_rate[:self.n_active_pool] - self.f_target)

        if instrument is not None:
            active_in_count = int(np.count_nonzero(spike_vec))
            instrument.adds += self.n_active_pool * self.fan_in
            instrument.sram_read_b += (self.n_active_pool * self.fan_in * 2) + active_in_count

        return top_k_indices
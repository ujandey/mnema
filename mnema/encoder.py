# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np

class EventEncoder:
    def __init__(self, encoder_type="ttfs", T=16, theta=0.05, eta_theta=1e-3, r_target=0.15):
        self.encoder_type = encoder_type
        self.T = T
        self.theta = theta
        self.eta_theta = eta_theta
        self.r_target = r_target
        self.x_ref = None

    def encode(self, x: np.ndarray, instrument=None) -> np.ndarray:
        """
        Encodes a 1D or 2D intensity array into binary event spikes [T, N_in].
        """
        x_flat = x.flatten().astype(np.float32)
        n_in = x_flat.shape[0]

        if self.encoder_type == "ttfs":
            # Normalize to [0, 1]
            x_norm = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min() + 1e-7)
            # Firing time t = ceil(T * (1 - x_norm))
            spike_times = np.ceil(self.T * (1.0 - x_norm)).astype(int)
            spike_times = np.clip(spike_times, 0, self.T - 1)

            spikes = np.zeros((self.T, n_in), dtype=np.uint8)
            for idx, t in enumerate(spike_times):
                if x_norm[idx] > self.theta:
                    spikes[t, idx] = 1

        elif self.encoder_type == "delta":
            if self.x_ref is None:
                self.x_ref = np.zeros_like(x_flat)
            
            diff = x_flat - self.x_ref
            spikes = np.zeros((1, n_in), dtype=np.uint8)
            on_events = diff > self.theta
            off_events = diff < -self.theta
            
            spikes[0, on_events | off_events] = 1
            self.x_ref[on_events | off_events] = x_flat[on_events | off_events]

        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        # Homeostatic rate adaptation
        measured_rate = np.mean(spikes)
        self.theta += self.eta_theta * (measured_rate - self.r_target)
        self.theta = max(1e-4, self.theta)

        if instrument is not None:
            instrument.spikes += int(np.sum(spikes))
            instrument.sram_read_b += n_in * 4

        return spikes
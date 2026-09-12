# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np

class BaselineMLP:
    def __init__(self, n_in=784, n_hidden=256, n_out=10, lr=0.01, use_replay=False, buffer_size=300):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        self.use_replay = use_replay
        self.buffer_size = buffer_size

        # He initialization
        self.W1 = np.random.randn(n_in, n_hidden).astype(np.float32) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = np.random.randn(n_hidden, n_out).astype(np.float32) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out, dtype=np.float32)

        # Experience Replay Buffer
        self.buffer_x = []
        self.buffer_y = []

    def forward(self, x: np.ndarray, instrument=None):
        # Layer 1: Dense MACs
        h_pre = np.dot(x, self.W1) + self.b1
        h = np.maximum(0, h_pre)  # ReLU
        
        # Layer 2: Dense MACs
        logits = np.dot(h, self.W2) + self.b2
        
        exp_vals = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

        if instrument is not None:
            batch_size = x.shape[0] if x.ndim > 1 else 1
            instrument.macs += batch_size * (self.n_in * self.n_hidden + self.n_hidden * self.n_out)
            instrument.sram_read_b += (self.n_in * self.n_hidden + self.n_hidden * self.n_out) * 4

        return h, probs

    def train_step(self, x: np.ndarray, y: int, is_replay: bool = False, instrument=None):
        x = x.reshape(1, -1)
        h, probs = self.forward(x, instrument=instrument)

        # Gradient calculation
        target_vec = np.zeros((1, self.n_out), dtype=np.float32)
        target_vec[0, y] = 1.0
        
        d_logits = (probs - target_vec)
        d_W2 = np.dot(h.T, d_logits)
        d_b2 = np.sum(d_logits, axis=0)

        d_h = np.dot(d_logits, self.W2.T) * (h > 0)
        d_W1 = np.dot(x.T, d_h)
        d_b1 = np.sum(d_h, axis=0)

        # Update weights
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

        if instrument is not None:
            instrument.macs += (self.n_hidden * self.n_out + self.n_in * self.n_hidden)
            instrument.sram_write_b += (self.W1.size + self.W2.size) * 4
            instrument.weight_updates += self.W1.size + self.W2.size

        # Buffer update and single rehearsal step for ReplayMLP
        if self.use_replay and not is_replay:
            if len(self.buffer_x) < self.buffer_size:
                self.buffer_x.append(x)
                self.buffer_y.append(y)
            else:
                idx = np.random.randint(0, self.buffer_size)
                self.buffer_x[idx] = x
                self.buffer_y[idx] = y

            # Sample a past item and train once without recursing
            rep_idx = np.random.randint(0, len(self.buffer_x))
            rx, ry = self.buffer_x[rep_idx], self.buffer_y[rep_idx]
            self.train_step(rx, ry, is_replay=True, instrument=instrument)
# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np

class ReadoutArbitrator:
    def __init__(self, a1=2.0, a2=1.0, a3=1.0):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def arbitrate(self, y_f: np.ndarray, conf_f: float, y_c: np.ndarray) -> tuple:
        """
        Combines Fast Store and Slow Cortex outputs based on confidence and familiarity.
        y_hat = softmax(alpha * y_f + (1 - alpha) * y_c)
        """
        familiarity_c = float(np.max(y_c) - np.mean(y_c))
        
        # Logistic gate for alpha
        logit = self.a1 * conf_f - self.a3 * familiarity_c
        alpha = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))

        # Weighted combination
        combined_logits = alpha * y_f + (1.0 - alpha) * y_c
        
        # Numerically stable softmax
        exp_vals = np.exp(combined_logits - np.max(combined_logits))
        probs = exp_vals / np.sum(exp_vals)
        pred = int(np.argmax(probs))

        return pred, probs, float(alpha)
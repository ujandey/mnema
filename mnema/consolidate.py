# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import numpy as np

class SleepConsolidator:
    def __init__(self, n_replay=64, prio_alpha=0.6, salience_decrement=2):
        self.n_replay = n_replay
        self.prio_alpha = prio_alpha
        self.salience_decrement = salience_decrement

    def consolidate(self, fast_store, slow_cortex, instrument=None):
        """
        Replays sparse codes directly from Fast Store into Slow Cortex.
        Consolidated traces have their fast salience decremented to recycle capacity.
        """
        active_rows = np.where(fast_store.s > 0)[0]
        if len(active_rows) == 0:
            return 0

        # Prioritized experience sampling based on salience and age
        saliences = fast_store.s[active_rows]
        probs = saliences ** self.prio_alpha
        probs /= probs.sum()

        sample_count = min(self.n_replay, len(active_rows))
        sampled_rows = np.random.choice(active_rows, size=sample_count, replace=False, p=probs)

        replayed = 0
        for row_idx in sampled_rows:
            target_class = int(np.argmax(fast_store.M[row_idx]))
            
            # Synthetic activation code from active row
            code = np.array([row_idx], dtype=np.int32)
            
            # Forward & local update in cortex
            slow_cortex.reset_state()
            slow_cortex.forward(code, instrument=instrument)
            slow_cortex.learn(target_class, instrument=instrument)

            # Decrement fast store salience (frees capacity for continuous streaming)
            fast_store.s[row_idx] = max(0.0, fast_store.s[row_idx] - self.salience_decrement)
            if fast_store.s[row_idx] == 0:
                fast_store.M[row_idx] = 0
            
            replayed += 1

        return replayed
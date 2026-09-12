# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

"""
Honest-audit probe: finds arithmetic that the EnergyInstrument does NOT charge for.

The headline claim "MNEMA executes 0 dense MACs" is measured by instrument counters
that are incremented manually at each call site. Any dense NumPy arithmetic without a
matching counter increment is invisible to the energy projection. This script locates
that gap and prices it, so the claim can be stated with the correct scope.
"""
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mnema.model import MNEMA
from instrument.counters import EnergyInstrument

CARD = "instrument/tech/asic_45nm.yaml"
MAC_J = 3.7e-12
SRAM_32B_J = 5.0e-12


def main(n=200):
    print("=== Uncounted-Arithmetic Audit (45nm ASIC card) ===\n")

    model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
    cx = model.cortex
    elig = cx.eligibility
    n_elig = elig.size
    elig_bytes = elig.nbytes

    # SlowCortex.forward() executes `self.eligibility *= self.rho_e` on every pass.
    # That is a dense elementwise multiply over the full [out_dim, in_features] trace,
    # plus a full read and full write of the array. None of it touches the instrument.
    mult_per_fwd = n_elig
    e_mult = mult_per_fwd * MAC_J
    e_traffic = (elig_bytes / 4.0) * SRAM_32B_J * 2  # read + write

    # SlowCortex.learn() additionally forms error_signal[:,None] * eligibility.
    mult_per_learn = n_elig

    print(f"cortex.eligibility shape      : {elig.shape}  ({n_elig:,} fp32 = {elig_bytes/1024:.0f} KB)")
    print(f"rho_e decay multiplies / fwd  : {mult_per_fwd:,}  (instrument charges 0)")
    print(f"delta_w multiplies / learn    : {mult_per_learn:,}  (instrument charges 0)\n")

    # Measure what the instrument DOES charge, inference-only.
    rng = np.random.default_rng(0)
    x = rng.random(784, dtype=np.float32)
    for i in range(50):
        model.step(rng.random(784, dtype=np.float32), y=i % 10, is_training=True)

    inst = EnergyInstrument(CARD)
    with inst:
        for _ in range(n):
            model.step(x, is_training=False, instrument=inst)
    counted = inst.project_energy()["energy_microjoules"] / n

    uncounted = (e_mult + e_traffic) * 1e6
    print(f"COUNTED   energy / inference  : {counted:.4f} uJ")
    print(f"UNCOUNTED eligibility decay   : {uncounted:.4f} uJ")
    print(f"  - dense fp32 multiplies     : {e_mult*1e6:.4f} uJ")
    print(f"  - SRAM read+write of trace  : {e_traffic*1e6:.4f} uJ")
    print(f"TRUE-COST estimate / inference: {counted + uncounted:.4f} uJ")
    print(f"Understatement factor         : {(counted + uncounted)/counted:.2f}x\n")

    print("Architectural note: the eligibility trace is only required to assign credit")
    print("during learning. Gating it behind `is_training` removes the entire cost from")
    print("the inference path and makes the 0-MAC inference claim exact.\n")

    # Confirm the trace is genuinely unused when not learning.
    before = cx.eligibility.copy()
    model.step(x, is_training=False)
    changed = not np.array_equal(before, cx.eligibility)
    print(f"Does eligibility mutate during is_training=False? {changed}  "
          f"<- {'BUG: yes, pure waste' if changed else 'clean'}")

    os.makedirs("results", exist_ok=True)
    with open("results/uncounted_audit.json", "w") as f:
        json.dump({
            "eligibility_elements": int(n_elig),
            "eligibility_kb": elig_bytes / 1024,
            "uncounted_multiplies_per_forward": int(mult_per_fwd),
            "counted_uj_per_inference": counted,
            "uncounted_uj_per_inference": uncounted,
            "true_cost_uj_per_inference": counted + uncounted,
            "understatement_factor": (counted + uncounted) / counted,
            "eligibility_mutates_in_inference": bool(changed),
        }, f, indent=2)
    print("\nSaved -> results/uncounted_audit.json")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mnema.model import MNEMA
from baselines.mlp import BaselineMLP
from instrument.counters import EnergyInstrument

CARD = "instrument/tech/asic_45nm.yaml"


def attribute(counts, card):
    """Splits projected energy into its physical contributors."""
    parts = {
        "compute: dense MACs": counts.macs * card.get("mac_fp32_J", 0.0),
        "compute: integer adds": counts.adds * card.get("add_int32_J", 0.0),
        "compute: sparse synops": counts.synops * card.get("synop_J", 0.0),
        "compute: neuron updates": counts.neuron_updates * card.get("neuron_update_J", 0.0),
        "data movement: SRAM reads": (counts.sram_read_b / 4.0) * card.get("sram_read_32b_J", 0.0),
        "data movement: DRAM reads": (counts.dram_read_b / 4.0) * card.get("dram_read_32b_J", 0.0),
    }
    total = sum(parts.values())
    return parts, total


def main(n=200):
    print("=== Energy Attribution: where the joules actually go (45nm ASIC card) ===\n")
    rng = np.random.default_rng(0)
    x = rng.random(784, dtype=np.float32)

    # --- MNEMA inference ---
    model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
    for i in range(50):
        model.step(rng.random(784, dtype=np.float32), y=i % 10, is_training=True)

    inst = EnergyInstrument(CARD)
    with inst:
        for _ in range(n):
            model.step(x, is_training=False, instrument=inst)
    mn_parts, mn_total = attribute(inst.counts, inst.tech_card)
    mn_spikes = inst.counts.spikes / n

    # --- MLP inference ---
    mlp = BaselineMLP()
    inst2 = EnergyInstrument(CARD)
    with inst2:
        for _ in range(n):
            mlp.forward(x.reshape(1, -1), instrument=inst2)
    ml_parts, ml_total = attribute(inst2.counts, inst2.tech_card)

    for label, parts, total, cnt in (("MNEMA", mn_parts, mn_total, inst.counts),
                                     ("MLP (dense)", ml_parts, ml_total, inst2.counts)):
        print(f"--- {label}: {total/n*1e6:.4f} uJ per inference ---")
        for k, v in parts.items():
            if v == 0:
                continue
            print(f"  {k:<32} {v/n*1e6:>9.4f} uJ  ({100*v/total:>5.1f}%)")
        print(f"  {'SRAM bytes touched / sample':<32} {cnt.sram_read_b/n:>9,.0f} B")
        print()

    print(f"MNEMA input events (spikes) per sample : {mn_spikes:.0f} of 784x16 raster slots "
          f"({100*mn_spikes/(784*16):.1f}% dense)")
    print(f"Energy ratio (MLP / MNEMA)             : {ml_total/mn_total:.2f}x")
    print(f"Data-movement share  MNEMA             : "
          f"{100*mn_parts['data movement: SRAM reads']/mn_total:.1f}%")
    print(f"Data-movement share  MLP               : "
          f"{100*ml_parts['data movement: SRAM reads']/ml_total:.1f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/energy_breakdown.json", "w") as f:
        json.dump({
            "n_inferences": n,
            "mnema_uj_per_inference": mn_total / n * 1e6,
            "mlp_uj_per_inference": ml_total / n * 1e6,
            "mnema_parts_uj": {k: v / n * 1e6 for k, v in mn_parts.items()},
            "mlp_parts_uj": {k: v / n * 1e6 for k, v in ml_parts.items()},
            "mnema_spikes_per_sample": mn_spikes,
        }, f, indent=2)
    print("\nSaved -> results/energy_breakdown.json")


if __name__ == "__main__":
    main()

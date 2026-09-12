# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import sys
import os
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mnema.model import MNEMA
from baselines.mlp import BaselineMLP
from instrument.counters import EnergyInstrument

KB = 1024.0


def mnema_footprint(m: MNEMA):
    """Exact byte-level static footprint of every persistent MNEMA array."""
    sep = m.separator
    st = m.store
    cx = m.cortex

    items = {
        "[S] separator.indices (frozen connectivity)": sep.indices.nbytes,
        "[S] separator.theta (homeostatic thresh)": sep.theta.nbytes,
        "[S] separator.running_rate": sep.running_rate.nbytes,
        "[F] store.M (associative weights, alloc)": st.M.nbytes,
        "[F] store.s (salience)": st.s.nbytes,
        "[F] store.tau (timestamps)": st.tau.nbytes,
        "[C] cortex.syn_fc.u (Benna-Fusi m=4)": cx.syn_fc.u.nbytes,
        "[C] cortex.eligibility (trace)": cx.eligibility.nbytes,
        "[C] cortex.v + cortex.b (neuron state)": cx.v.nbytes + cx.b.nbytes,
    }
    return items


def mlp_footprint(m: BaselineMLP, replay_samples: int, input_dim: int = 784):
    items = {
        "W1 (784x256 fp32)": m.W1.nbytes,
        "b1": m.b1.nbytes,
        "W2 (256x10 fp32)": m.W2.nbytes,
        "b2": m.b2.nbytes,
    }
    if m.use_replay:
        items[f"replay buffer ({replay_samples} raw samples fp32)"] = replay_samples * input_dim * 4
    return items


def latency_and_energy_per_sample(n=200):
    """Per-sample wall-clock and projected-energy cost, inference only."""
    rng = np.random.default_rng(0)
    x = rng.random(784, dtype=np.float32)

    model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
    for i in range(50):  # warm the store so reads are realistic
        model.step(rng.random(784, dtype=np.float32), y=i % 10, is_training=True)

    inst = EnergyInstrument("instrument/tech/asic_45nm.yaml")
    t0 = time.perf_counter()
    with inst:
        for _ in range(n):
            model.step(x, is_training=False, instrument=inst)
    mnema_wall = (time.perf_counter() - t0) / n
    mnema_e = inst.project_energy()["energy_microjoules"] / n
    mnema_ops = dict(macs=inst.counts.macs // n, adds=inst.counts.adds // n,
                     synops=inst.counts.synops // n, spikes=inst.counts.spikes // n)

    mlp = BaselineMLP()
    inst2 = EnergyInstrument("instrument/tech/asic_45nm.yaml")
    t0 = time.perf_counter()
    with inst2:
        for _ in range(n):
            mlp.forward(x.reshape(1, -1), instrument=inst2)
    mlp_wall = (time.perf_counter() - t0) / n
    mlp_e = inst2.project_energy()["energy_microjoules"] / n
    mlp_ops = dict(macs=inst2.counts.macs // n, adds=inst2.counts.adds // n,
                   synops=inst2.counts.synops // n, spikes=inst2.counts.spikes // n)

    return {
        "MNEMA": {"wall_ms": mnema_wall * 1e3, "energy_uj": mnema_e, "ops": mnema_ops},
        "MLP": {"wall_ms": mlp_wall * 1e3, "energy_uj": mlp_e, "ops": mlp_ops},
    }


def energy_across_tech_cards():
    """Same MNEMA op-counts projected onto all three technology cards."""
    rng = np.random.default_rng(0)
    model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
    out = {}
    for card in ("asic_45nm", "loihi2", "odin_28nm"):
        inst = EnergyInstrument(f"instrument/tech/{card}.yaml")
        with inst:
            for _ in range(100):
                model.step(rng.random(784, dtype=np.float32), is_training=False, instrument=inst)
        p = inst.project_energy()
        out[card] = {
            "name": inst.tech_card.get("name"),
            "source": p["tech_source"],
            "uj_per_sample": p["energy_microjoules"] / 100,
        }
    return out


def main():
    print("=== MNEMA Static Memory Footprint & Per-Sample Cost Audit ===\n")

    m = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
    mn = mnema_footprint(m)
    mn_total = sum(mn.values())

    print("--- MNEMA static footprint (default config) ---")
    for k, v in mn.items():
        print(f"  {k:<45} {v/KB:>10.1f} KB")
    print(f"  {'TOTAL RESIDENT':<45} {mn_total/KB:>10.1f} KB")
    print(f"  {'of which LEARNED-CONTENT budget [F]':<45} {m.store.budget_bytes/KB:>10.1f} KB (hard cap, task-invariant)")
    print(f"  {'of which FIXED/frozen scaffold':<45} "
          f"{(mn['[S] separator.indices (frozen connectivity)'])/KB:>10.1f} KB (random, never learned)\n")

    naive = BaselineMLP(use_replay=False)
    rep = BaselineMLP(use_replay=True, buffer_size=300)
    for label, mdl, nrep in (("MLP (Naive)", naive, 0), ("MLP (+Replay, 300)", rep, 300)):
        f = mlp_footprint(mdl, nrep)
        print(f"--- {label} static footprint ---")
        for k, v in f.items():
            print(f"  {k:<45} {v/KB:>10.1f} KB")
        print(f"  {'TOTAL RESIDENT':<45} {sum(f.values())/KB:>10.1f} KB\n")

    print("--- Per-sample inference cost ---")
    lat = latency_and_energy_per_sample(200)
    for k, v in lat.items():
        print(f"  {k:<8} wall {v['wall_ms']:>8.3f} ms | projected {v['energy_uj']:>8.3f} uJ | "
              f"MACs {v['ops']['macs']:>8,} | adds {v['ops']['adds']:>10,} | synops {v['ops']['synops']:>7,}")
    print()

    print("--- Same MNEMA workload across technology cards (100 inferences) ---")
    tech = energy_across_tech_cards()
    for k, v in tech.items():
        print(f"  {v['name']:<28} {v['uj_per_sample']:>9.3f} uJ/sample   [{v['source']}]")

    os.makedirs("results", exist_ok=True)
    with open("results/footprint_results.json", "w") as f:
        json.dump({
            "mnema_footprint_bytes": mn,
            "mnema_total_bytes": mn_total,
            "mlp_naive_footprint_bytes": mlp_footprint(naive, 0),
            "mlp_replay_footprint_bytes": mlp_footprint(rep, 300),
            "per_sample": lat,
            "tech_cards": tech,
        }, f, indent=2)
    print("\nSaved -> results/footprint_results.json")


if __name__ == "__main__":
    main()

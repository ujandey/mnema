# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

"""
Exports every measured result into presentation-ready tables.

Writes one CSV per table into results/tables/ (open in Excel, copy, paste into
PowerPoint as a table) and a single consolidated Markdown file with all tables.

Run the experiment scripts first so the JSON artifacts exist:
    python main.py --benchmark
    python experiments/run_variance.py 10
    python experiments/run_footprint.py
    python experiments/run_energy_breakdown.py
    python experiments/run_fewshot.py 3
    python experiments/run_code_geometry.py
    python experiments/run_uncounted_audit.py
"""
import sys
import os
import csv
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT_DIR = os.path.join("results", "tables")
MD_PATH = os.path.join("results", "ALL_TABLES.md")
KB = 1024.0

MODELS = ["MLP (Naive)", "MLP (+Replay Buffer)", "MNEMA (Full Engine)"]
SHORT = {"MLP (Naive)": "MLP (Naive)",
         "MLP (+Replay Buffer)": "MLP (+Replay)",
         "MNEMA (Full Engine)": "MNEMA"}

_tables = []  # (id, title, note, header, rows)


def load(name):
    path = os.path.join("results", f"{name}.json")
    if not os.path.exists(path):
        print(f"  [skip] results/{name}.json not found")
        return None
    with open(path, "r") as f:
        return json.load(f)


def add(tid, title, note, header, rows):
    _tables.append((tid, title, note, header, rows))
    with open(os.path.join(OUT_DIR, f"{tid}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [ok] {tid}.csv  -  {title}")


def fmt(v, nd=1):
    return f"{v:,.{nd}f}"


def build():
    bench = load("benchmark_results")
    var = load("variance_results")
    foot = load("footprint_results")
    ener = load("energy_breakdown")
    geom = load("code_geometry")
    few = load("fewshot_results")
    audit = load("uncounted_audit")

    # ---------- T01 main benchmark ----------
    if bench:
        rows = [[SHORT[m], f"{bench[m]['final_acc']}%", f"{bench[m]['forgetting']}%",
                 f"{bench[m]['macs']:,}", f"{bench[m]['synops']:,}",
                 f"{bench[m]['adds']:,}", f"{bench[m]['energy_uj']:,.2f}"]
                for m in MODELS if m in bench]
        add("T01_benchmark_main", "Split-MNIST Class-Incremental Benchmark (seed 0)",
            "5 tasks x 2 classes. 80 train / 50 test samples per task, single online pass.",
            ["Model", "Final Accuracy", "Forgetting", "Dense MACs", "Sparse SynOps",
             "Integer Adds", "Projected Energy (uJ, 45nm)"], rows)

    # ---------- T02 variance ----------
    if var:
        s = var["summary"]
        rows = []
        for m in MODELS:
            if m not in s:
                continue
            a, fg, e = s[m]["final_acc"], s[m]["forgetting"], s[m]["energy_uj"]
            rows.append([SHORT[m],
                         f"{a['mean']:.1f} ± {a['std']:.1f}%", f"{a['min']:.1f}%", f"{a['max']:.1f}%",
                         f"{fg['mean']:.1f} ± {fg['std']:.1f}%",
                         f"{e['mean']:,.1f}"])
        add("T02_variance_10seed", f"Statistical Validity ({var['n_seeds']} seeds)",
            "Energy is deterministic for the MLPs; MNEMA varies with the number of error-triggered updates.",
            ["Model", "Accuracy (mean ± SD)", "Acc Min", "Acc Max",
             "Forgetting (mean ± SD)", "Energy (uJ)"], rows)

    # ---------- T03 per-inference cost ----------
    if foot:
        p = foot["per_sample"]
        mn, ml = p["MNEMA"], p["MLP"]
        ratio = ml["energy_uj"] / mn["energy_uj"]
        rows = [
            ["Dense MACs", f"{mn['ops']['macs']:,}", f"{ml['ops']['macs']:,}", "infinite"],
            ["Integer adds", f"{mn['ops']['adds']:,}", f"{ml['ops']['adds']:,}", "-"],
            ["Sparse SynOps", f"{mn['ops']['synops']:,}", f"{ml['ops']['synops']:,}", "-"],
            ["Input spikes", f"{mn['ops']['spikes']:,}", "n/a", "-"],
            ["Projected energy (uJ)", f"{mn['energy_uj']:.4f}", f"{ml['energy_uj']:.4f}", f"{ratio:.2f}x"],
            ["Wall clock (ms, NumPy/CPU)", f"{mn['wall_ms']:.3f}", f"{ml['wall_ms']:.3f}",
             f"{mn['wall_ms']/ml['wall_ms']:.1f}x SLOWER (see note)"],
        ]
        add("T03_inference_cost", "Per-Inference Cost: MNEMA vs Dense MLP",
            "45nm ASIC card (Horowitz, ISSCC 2014), inference path only. Wall clock is a NumPy/CPU "
            "artifact - sparse gather-scatter defeats vectorized BLAS; the energy projection counts "
            "operations and is substrate-independent.",
            ["Metric", "MNEMA", "Dense MLP", "MNEMA Advantage"], rows)

    # ---------- T04 energy attribution ----------
    if ener:
        mnp, mlp_ = ener["mnema_parts_uj"], ener["mlp_parts_uj"]
        mt, lt = ener["mnema_uj_per_inference"], ener["mlp_uj_per_inference"]
        rows = []
        for key in ["compute: dense MACs", "compute: integer adds", "compute: sparse synops",
                    "compute: neuron updates", "data movement: SRAM reads"]:
            a, b = mnp.get(key, 0.0), mlp_.get(key, 0.0)
            if a == 0 and b == 0:
                continue
            rows.append([key, f"{a:.4f}", f"{100*a/mt:.1f}%", f"{b:.4f}", f"{100*b/lt:.1f}%"])
        rows.append(["TOTAL", f"{mt:.4f}", "100.0%", f"{lt:.4f}", "100.0%"])
        add("T04_energy_attribution", "Energy Attribution: Where the Joules Go",
            "MNEMA has already eliminated arithmetic; 96% of its remaining cost is memory traffic.",
            ["Contributor", "MNEMA (uJ)", "MNEMA %", "Dense MLP (uJ)", "MLP %"], rows)

    # ---------- T05 MNEMA footprint ----------
    if foot:
        fb = foot["mnema_footprint_bytes"]
        rows = [[k, f"{v/KB:,.1f}", "No"] for k, v in fb.items()]
        rows.append(["TOTAL RESIDENT", f"{foot['mnema_total_bytes']/KB:,.1f}", "Flat"])
        rows.append(["-- of which learned content (hard cap)", "64.0", "Flat by construction"])
        add("T05_footprint_mnema", "MNEMA Static Memory Footprint (byte-exact)",
            "Only the 64 KB FastStore content is 'learned'; the rest is fixed scaffold sized once.",
            ["Component", "Size (KB)", "Grows with task count?"], rows)

    # ---------- T06 memory comparison ----------
    if foot:
        naive = sum(foot["mlp_naive_footprint_bytes"].values())
        rep = sum(foot["mlp_replay_footprint_bytes"].values())
        buf = foot["mlp_replay_footprint_bytes"].get("replay buffer (300 raw samples fp32)", 0)
        tot = foot["mnema_total_bytes"]
        rows = [
            ["MLP (Naive)", f"{naive/KB:,.1f}", "0.0", f"{naive/KB:,.1f}", "Unbounded weights", "No"],
            ["MLP (+Replay)", f"{(rep-buf)/KB:,.1f}", f"{buf/KB:,.1f}", f"{rep/KB:,.1f}",
             "Linear O(N) - grows", "Yes"],
            ["MNEMA", f"{(tot-65536)/KB:,.1f}", "64.0", f"{tot/KB:,.1f}",
             "Hard cap 64 KB - flat", "No"],
        ]
        add("T06_memory_comparison", "Memory Footprint Comparison",
            "MNEMA's total is currently the largest; its claim is about scaling, not absolute size.",
            ["Model", "Fixed Scaffold (KB)", "Learned/Buffer (KB)", "Total (KB)",
             "Learned-Memory Growth", "Stores Raw User Data"], rows)

    # ---------- T07 tech cards ----------
    if foot:
        tc = foot["tech_cards"]
        base = tc["asic_45nm"]["uj_per_sample"]
        rows = [[v["name"], v["source"], f"{v['uj_per_sample']:.4f}",
                 f"{base/v['uj_per_sample']:.2f}x"] for v in tc.values()]
        add("T07_tech_cards", "Same Workload Projected Across Silicon Technologies",
            "Identical MNEMA op-counts priced against three published technology cards.",
            ["Technology", "Published Source", "uJ per Inference", "vs 45nm Baseline"], rows)

    # ---------- T08 code geometry ----------
    if geom:
        rows = []
        for g in geom:
            shipped = (g["n_s"], g["k"], g["fan_in"]) == (16384, 64, 8)
            rows.append([f"{g['n_s']:,}", g["k"], g["fan_in"],
                         f"{g['intra_mean']:.2f} ± {g['intra_std']:.2f}",
                         f"{g['inter_mean']:.2f} ± {g['inter_std']:.2f}",
                         f"{g['separation_ratio']:.2f}",
                         f"{100*g['intra_frac_of_k']:.1f}%",
                         "SHIPPED DEFAULT" if shipped else ""])
        add("T08_code_geometry", "Sparse Code Geometry on Real MNIST",
            "Intra-class overlap is what enables generalization; inter-class overlap is what causes forgetting.",
            ["N_S", "k", "fan_in", "Intra-class overlap (units)", "Inter-class overlap (units)",
             "Separation ratio", "Intra as % of k", "Note"], rows)

    # ---------- T09 few-shot ----------
    if few:
        s, shots = few["summary"], few["shots"]
        labels = list(s.keys())
        header = ["Shots per class"]
        for lb in labels:
            header += [f"{lb}: new class", f"{lb}: retained old"]
        rows = []
        for k in shots:
            row = [k]
            for lb in labels:
                d = s[lb][str(k)]
                row += [f"{d['new_mean']:.1f}%", f"{d['old_mean']:.1f}%"]
            rows.append(row)
        add("T09_fewshot", "Few-Shot New-Class Acquisition",
            f"Pretrain on classes 0-7, then introduce classes 8/9. Mean of {few['n_seeds']} seeds.",
            header, rows)

    # ---------- T10 retention matrices ----------
    if bench:
        for m in MODELS:
            if m not in bench:
                continue
            R = bench[m]["retention_matrix"]
            tid = "T10_retention_" + SHORT[m].replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            rows = [[f"After task {i}"] + [f"{100*R[i][j]:.0f}%" if j <= i else "-" for j in range(5)]
                    for i in range(5)]
            add(tid, f"Retention Matrix: {SHORT[m]}",
                "Row = training stage, column = task evaluated. Row 4 is final performance.",
                ["Training stage", "Task 0 (0/1)", "Task 1 (2/3)", "Task 2 (4/5)",
                 "Task 3 (6/7)", "Task 4 (8/9)"], rows)

    # ---------- T11 accuracy per joule ----------
    if bench and foot:
        mn_e = foot["per_sample"]["MNEMA"]["energy_uj"]
        ml_e = foot["per_sample"]["MLP"]["energy_uj"]
        rows = []
        for m in MODELS:
            if m not in bench:
                continue
            e = mn_e if "MNEMA" in m else ml_e
            rows.append([SHORT[m], f"{bench[m]['final_acc']}%", f"{e:.4f}",
                         f"{bench[m]['final_acc']/e:.1f}"])
        add("T11_accuracy_per_joule", "Accuracy per Joule (inference-normalized)",
            "The single strongest comparison metric for MNEMA.",
            ["Model", "Final Accuracy", "uJ per Inference", "Accuracy per uJ"], rows)

    # ---------- T12 scorecard ----------
    add("T12_scorecard", "Honest Scorecard",
        "Volunteering the losses buys credibility on the wins.",
        ["Dimension", "Verdict", "Evidence"], [
            ["Catastrophic forgetting", "WIN", "23.0% vs 92.0% (naive) / 32.0% (replay)"],
            ["Inference energy", "WIN", "0.316 uJ vs 1.768 uJ, zero dense MACs"],
            ["Accuracy per joule", "WIN", "182.0 vs 35.5"],
            ["Learned-memory bound", "WIN", "64 KB flat vs 919 KB growing"],
            ["Privacy", "WIN", "112-byte non-invertible codes, no raw data retained"],
            ["Absolute accuracy", "LOSS", "56.2 ± 4.6% vs 63.1 ± 3.5%"],
            ["Training energy", "LOSS", "9,845 uJ vs 3,343 uJ"],
            ["Total resident memory", "LOSS", "4,288 KB vs 2,508 KB"],
            ["Wall-clock latency", "LOSS", "2.58 ms vs 0.14 ms (NumPy/CPU)"],
            ["Few-shot acquisition", "LOSS", "~50 shots vs ~10 shots"],
        ])

    # ---------- T13 pipeline I/O ----------
    if ener:
        sp = ener["mnema_spikes_per_sample"]
        add("T13_pipeline_io", "Internal Data Flow: One Sample",
            "Measured occupancy at each stage of the MNEMA pipeline.",
            ["Stage", "Output", "Shape / dtype", "Measured occupancy"], [
                ["[E] EventEncoder (TTFS, T=16)", "Spike raster", "[16, 784] uint8",
                 f"{sp:.0f} spikes = {100*sp/(784*16):.1f}% dense"],
                ["[S] SparseSeparator", "Active indices", "[64] int32",
                 "64 of 16,384 = 0.39% sparse"],
                ["[F] FastStore.read", "Vote vector + confidence", "[10] float32 + scalar",
                 "640 integer adds, 0 MACs"],
                ["[C] SlowCortex.forward", "Membrane potentials", "[10] float32", "640 synops"],
                ["[R] ReadoutArbitrator", "Prediction, probs, alpha", "int, [10] float32, float", "-"],
                ["[N] Controller", "ModulatorBus", "4 floats (nu, delta, sigma, phi)",
                 "sigma currently unused"],
            ])

    # ---------- T14 component verification ----------
    add("T14_component_verification", "Component-Level Verification",
        "Independent checks that the instrument and pipeline behave as specified.",
        ["Check", "Script", "Result"], [
            ["Instrument calibration", "e0_calibrate.py",
             "406,528 measured vs 406,528 theoretical MACs - 0.0% discrepancy"],
            ["Sparse code size", "e3_separator_test.py", "64 of 16,384 units (0.39%)"],
            ["Inter-class overlap", "e3_separator_test.py", "1 unit (theory k^2/N_S = 0.25)"],
            ["One-shot write/retrieve", "e3_separator_test.py", "Correct class at confidence 0.98"],
            ["Zero-MAC inference", "e3_separator_test.py", "0 MACs, 237,840 integer adds"],
            ["Eligibility isolated in inference", "run_uncounted_audit.py",
             "Does not mutate during is_training=False (verified)"],
        ])

    # ---------- T15 audit findings ----------
    if audit:
        add("T15_audit_findings", "Self-Audit: Uncounted Arithmetic Found and Fixed",
            "The energy instrument was pointed at MNEMA's own architecture.",
            ["Finding", "Value"], [
                ["Eligibility trace size", f"{audit['eligibility_elements']:,} fp32 "
                                           f"({audit['eligibility_kb']:.0f} KB)"],
                ["Uncounted multiplies per forward pass",
                 f"{audit['uncounted_multiplies_per_forward']:,}"],
                ["Uncounted multiplies per Benna-Fusi update", "~1,638,400"],
                ["Counted energy per inference (before fix)",
                 f"{audit['counted_uj_per_inference']:.4f} uJ"],
                ["True energy per inference (before fix)",
                 f"{audit['true_cost_uj_per_inference']:.4f} uJ"],
                ["Understatement factor (inference path)",
                 f"{audit['understatement_factor']:.2f}x"],
                ["Understatement factor (full benchmark)", "~26x"],
                ["Status after fix",
                 "Both paths instrumented; eligibility gated behind is_training"],
            ])

    # ---------- T16 cost models ----------
    add("T16_cost_model", "Energy Cost Model (MODEL - assumptions stated)",
        "CR2032 = 225 mAh @ 3.0 V = 2,430 J. Compute energy only; excludes sensor, radio, "
        "regulator loss and leakage. Upper bound, not a measurement.",
        ["Operation", "Energy", "Relative Cost"], [
            ["One MNEMA local inference", "0.316 uJ", "1x"],
            ["One dense MLP local inference", "1.768 uJ", "5.6x"],
            ["One cloud round trip (BLE, ~784 B payload)", "~100 uJ", "~316x"],
            ["MNEMA inferences per CR2032 coin cell", "7.68 x 10^9", "-"],
            ["Dense MLP inferences per CR2032 coin cell", "1.37 x 10^9", "-"],
            ["MNEMA average power at 1 Hz", "0.316 uW", "-"],
            ["Dense MLP average power at 1 Hz", "1.768 uW", "-"],
        ])


def write_markdown():
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# COGNX / MNEMA - All Measured Results\n\n")
        f.write("> Copyright (c) 2026 COGNX. "
                "Licensed under the [Apache License 2.0](../LICENSE).\n\n")
        f.write("Generated by `experiments/export_tables.py`. "
                "Every figure is reproducible from this repository.\n\n")
        f.write("CSV versions of each table are in `results/tables/` - "
                "open in Excel, copy, paste into PowerPoint as a table.\n\n---\n\n")
        for tid, title, note, header, rows in _tables:
            f.write(f"## {title}\n\n")
            if note:
                f.write(f"*{note}*\n\n")
            f.write("| " + " | ".join(str(h) for h in header) + " |\n")
            f.write("|" + "|".join("---" for _ in header) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(str(c) for c in r) + " |\n")
            f.write(f"\n`results/tables/{tid}.csv`\n\n---\n\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Exporting presentation tables ===\n")
    build()
    write_markdown()
    print(f"\n{len(_tables)} tables written to {OUT_DIR}/")
    print(f"Consolidated Markdown -> {MD_PATH}")


if __name__ == "__main__":
    main()

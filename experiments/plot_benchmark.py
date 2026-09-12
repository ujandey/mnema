# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

def generate_benchmark_figures():
    with open("results/benchmark_results.json", "r") as f:
        results = json.load(f)

    os.makedirs("results", exist_ok=True)
    models = list(results.keys())

    # --- Figure 1: Retention Heatmaps (R[t, j]) ---
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.5), sharey=True)
    
    for idx, (name, data) in enumerate(results.items()):
        R = np.array(data["retention_matrix"]) * 100.0
        ax = axes[idx]
        im = ax.imshow(R, cmap="RdYlGn", vmin=0, vmax=100)
        
        ax.set_title(f"{name}\nACC: {data['final_acc']}% | FM: {data['forgetting']}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluated Task", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Trained Up To Task", fontsize=10)
        
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"T{i}" for i in range(5)])
        ax.set_yticklabels([f"T{i}" for i in range(5)])

        # Annotate cell values
        for i in range(5):
            for j in range(5):
                if j <= i:
                    val = R[i, j]
                    color = "white" if val < 30 or val > 75 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Task Accuracy (%)")
    plt.suptitle("Class-Incremental Split-MNIST: Retention Matrix Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig("results/retention_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Figure 2: Energy vs Accuracy Frontier ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    for idx, (name, data) in enumerate(results.items()):
        acc = data["final_acc"]
        energy = data["energy_uj"]
        ax.scatter(energy, acc, color=colors[idx], s=180, zorder=5, label=name)
        ax.annotate(f"{name}\n({acc}%, {energy:,.1f} uJ)", 
                    (energy, acc), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha="center", 
                    fontsize=9, 
                    fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Projected Energy per Stream (uJ) — Log Scale [ASIC 45nm]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Class-IL Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Continual Learning Efficiency Frontier (Split-MNIST)", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.savefig("results/energy_accuracy_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("=== [PASS] Plots generated: results/retention_matrices.png & results/energy_accuracy_frontier.png ===")

if __name__ == "__main__":
    generate_benchmark_figures()
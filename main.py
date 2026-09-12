# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import argparse

def main():
    parser = argparse.ArgumentParser(description="MNEMA: Neuromorphic Continual Learning Engine")
    parser.add_argument("--benchmark", action="store_true", help="Run Split-MNIST Class-IL benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate benchmark evaluation plots")
    
    args = parser.parse_args()

    if args.benchmark:
        from experiments.run_split_mnist_benchmark import run_benchmark
        run_benchmark()
    elif args.plot:
        from experiments.plot_benchmark import generate_benchmark_figures
        generate_benchmark_figures()
    else:
        print("Usage:")
        print("  uv run python main.py --benchmark  # Run Split-MNIST benchmark")
        print("  uv run python main.py --plot       # Generate evaluation figures")

if __name__ == "__main__":
    main()

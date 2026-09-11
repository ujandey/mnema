"""Sequential, resumable full Split-MNIST Class-IL; use --quick on small machines."""
import os
# Avoid oversubscribed BLAS and excessive thread workspace on small CPUs.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
import yaml
from baselines.research_mlp import ResearchMLP, normalize
from baselines.research_replay import ReplayMLP
from baselines.ewc import EWC
from baselines.derpp import DERPP
from benchmarks.research_split_mnist import load_dataset, task_indices, sequence_hash
from experiments.research_memory import memory_report, faststore_memory
from experiments.research_state import evaluate
from experiments.research_results import atomic_json, digest, metrics, seal_result, valid_result, aggregate
from instrument.counters import EnergyInstrument
from mnema.model import MNEMA

METHODS = {"Naive MLP": "naive", "Replay-300 (Research)": "replay300",
           "Replay-64KiB": "replay64kib", "EWC": "ewc", "DER++-300": "derpp300", "MNEMA": "mnema"}
MNEMA_PARAMETERS = {"n_in": 784, "n_s": 16384, "k": 64, "d": 10, "budget_bytes": 65536}
CARD = ROOT / "instrument/tech/asic_45nm.yaml"


def build_model(method, seed, config):
    np.random.seed(seed)
    random.seed(seed)
    hp = config["hyperparameters"]
    if method == "MNEMA":
        return MNEMA(**MNEMA_PARAMETERS)
    if method == "Naive MLP":
        return ResearchMLP(lr=hp["lr"])
    if method.startswith("Replay"):
        return ReplayMLP(lr=hp["lr"], seed=seed + 10000,
                         budget_bytes=65536 if method == "Replay-64KiB" else None)
    if method == "EWC":
        return EWC(lr=hp["lr"], ewc_lambda=hp["ewc_lambda"])
    return DERPP(lr=hp["lr"], seed=seed + 10000, alpha=hp["der_alpha"], beta=hp["der_beta"])


def source_manifest():
    paths = set()
    for folder in ("mnema", "instrument", "baselines", "benchmarks"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.update((ROOT / "instrument/tech").glob("*.yaml"))
    for pattern in ("research_*.py", "*research_benchmark.py"):
        paths.update((ROOT / "experiments").glob(pattern))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def mnema_defaults():
    # Capture every scalar constructor setting plus array shapes/dtypes, not learned values.
    rng = np.random.get_state()
    np.random.seed(0)
    model = MNEMA(**MNEMA_PARAMETERS)
    def describe(value):
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "__dict__"):
            return {k: describe(v) for k, v in vars(value).items()}
        return value
    result = describe(model)
    del model
    np.random.set_state(rng)
    return result


def run_one(data, trains, tests, seed, method, config):
    model = build_model(method, seed, config)
    is_mnema = method == "MNEMA"
    initial_memory = memory_report(model, is_mnema)
    train_instrument = EnergyInstrument(str(CARD)) if is_mnema else None
    eval_instrument = EnergyInstrument(str(CARD))
    retention, checks, memory, predictions_saved = [], [], [], []
    monitor = {"observations": 0, "over_budget_observations": 0,
               "max_actual_active_payload_bytes": 0, "max_implementation_occupancy_bytes": 0}
    payload_sum = payload_square_sum = 0
    fisher_counts = []
    started = time.perf_counter()
    for task, order in enumerate(trains):
        print(f"  seed {seed} | {method} | task {task + 1}/5 | train={len(order)}", flush=True)
        for index in order:
            raw, label = data["train_img"][index], int(data["train_lbl"][index])
            if is_mnema:
                model.step(normalize(raw), y=label, is_training=True, instrument=train_instrument)
                current = faststore_memory(model.store)
                payload_sum += current["actual_active_payload_bytes"]
                payload_square_sum += current["actual_active_payload_bytes"] ** 2
                monitor["observations"] += 1
                monitor["over_budget_observations"] += int(current["actual_exceeds_configured_budget"])
                monitor["max_actual_active_payload_bytes"] = max(monitor["max_actual_active_payload_bytes"], current["actual_active_payload_bytes"])
                monitor["max_implementation_occupancy_bytes"] = max(monitor["max_implementation_occupancy_bytes"], current["implementation_occupancy_bytes"])
            else:
                model.observe(raw, label)
        if method == "EWC" and task < 4:
            limit = config["hyperparameters"]["fisher_samples"]
            # First N of the already shuffled TRAIN stream: uniform subset, no test tuning.
            fisher_indices = order if limit == 0 else order[:limit]
            model.consolidate(data["train_img"], data["train_lbl"], fisher_indices)
            fisher_counts.append(len(fisher_indices))
        row, task_predictions = [], []
        for evaluation_indices in tests:
            predicted, check = evaluate(model, data["test_img"], evaluation_indices,
                                        is_mnema, eval_instrument,
                                        reverse_check=is_mnema and task == 4)
            row.append(float(np.mean(predicted == data["test_lbl"][evaluation_indices])))
            checks.append(check)
            task_predictions.append(predicted.tolist())
        retention.append(row)
        predictions_saved.append(task_predictions)
        memory.append(memory_report(model, is_mnema))
    total_eval = 5 * sum(len(indices) for indices in tests)
    if is_mnema:
        count = monitor["observations"]
        monitor["mean_actual_active_payload_bytes"] = payload_sum / count
        monitor["std_actual_active_payload_bytes"] = float(np.sqrt(max(0, payload_square_sum / count - (payload_sum / count) ** 2)))
    result = {"status": "complete", "mode": config["mode"], "label": config["label"],
              "config_id": config["config_id"], "seed": seed, "method": method,
              "stream_hashes": [sequence_hash(i) for i in trains],
              "test_hashes": [sequence_hash(i) for i in tests],
              "retention_matrix": retention, "predictions": predictions_saved,
              "metrics": metrics(retention), "memory_initial": initial_memory,
              "memory_final": memory[-1], "memory_by_task": memory,
              "evaluation_checks": checks, "faststore_monitor": monitor if is_mnema else None,
              "training_samples": sum(map(len, trains)), "evaluation_samples": total_eval,
              "optimizer_updates": getattr(model, "updates", None),
              "replay_draws": getattr(model, "replay_draws", None), "fisher_sample_counts": fisher_counts,
              "wall_time_s": time.perf_counter() - started,
              "energy": {"is_measured": False, "cross_method_training_comparison_valid": False,
                         "inference_counts": asdict(eval_instrument.counts),
                         "inference_projected_uj_per_sample": eval_instrument.project_energy()["energy_microjoules"] / total_eval,
                         "native_mnema_training_counts": asdict(train_instrument.counts) if is_mnema else None,
                         "native_mnema_training_projected_uj": train_instrument.project_energy()["energy_microjoules"] if is_mnema else None,
                         "caveat": "Partial native counters; excludes checkpoint restoration, buffer/Fisher work and other uninstrumented operations. Not physical measurements or comparable training energy."}}
    del model
    gc.collect()
    return seal_result(result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default=None)
    parser.add_argument("--quick", action="store_true", help="Alias for --mode quick (debug only)")
    parser.add_argument("--seeds", type=int, default=None, help="Seeds 0,...,N-1 (default: quick 1, full 10)")
    parser.add_argument("--seed", type=int, default=None, help="Run one seed only; requires --method")
    parser.add_argument("--method", choices=tuple(METHODS.values()), default=None,
                        help="Run one method slug only; requires --seed")
    parser.add_argument("--quick-train", type=int, default=100)
    parser.add_argument("--quick-test", type=int, default=100)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--fisher-samples", type=int, default=0, help="Per-task training samples; 0 means all; no final-task Fisher")
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--der-beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="Archive this mode's previous outputs and rerun all pairs")
    args = parser.parse_args(argv)
    if args.quick and args.mode == "full":
        parser.error("--quick and --mode full conflict")
    args.mode = "quick" if args.quick else (args.mode or "full")
    args.seeds = args.seeds if args.seeds is not None else (1 if args.mode == "quick" else 10)
    if (args.seed is None) != (args.method is None):
        parser.error("--seed and --method must be provided together")
    if args.seed is not None and not 0 <= args.seed < args.seeds:
        parser.error("--seed must be in the configured seed range")
    if args.seeds < 1 or args.fisher_samples < 0 or args.lr <= 0:
        parser.error("Invalid seed count, Fisher sample count or learning rate")
    if any(not np.isfinite(v) or v < 0 for v in (args.ewc_lambda, args.der_alpha, args.der_beta)) or not np.isfinite(args.lr):
        parser.error("Coefficients must be finite and nonnegative")
    if args.mode == "quick" and not (1 <= args.quick_train <= 500 and 1 <= args.quick_test <= 200):
        parser.error("Quick mode permits 1-500 train and 1-200 test examples per task")
    return args


def selected_method(slug):
    """Resolve the stable CLI slug to the existing display name."""
    return next(method for method, value in METHODS.items() if value == slug)


def build_config(args, data):
    """Build the same scientific configuration for local and distributed runs."""
    _, _, sizes = task_indices(data, 0, args.mode == "quick", args.quick_train, args.quick_test)
    for item in sizes:
        label = "".join(map(str, item["classes"]))
        print(f"Task {label}: full train={item['full_train']}, full test={item['full_test']}; used train={item['train']}, test={item['test']}")
    config = {"schema_version": 1, "mode": args.mode,
              "label": "DEBUG RESULTS ONLY" if args.mode == "quick" else "FULL DATA / RESEARCH PROTOCOL",
              "dataset": "MNIST", "dataset_sha256": data["hashes"], "task_sizes": sizes,
              "seeds": list(range(args.seeds)), "methods": list(METHODS),
              "hyperparameters": {"lr": args.lr, "mlp_dimensions": [784, 256, 10], "mlp_dtype": "float32",
                                  "epochs": 1, "current_batch_size": 1, "replay_batch_size": 1,
                                  "replay_policy": "reservoir; sample before insertion; mean current+replay CE",
                                  "replay_capacity": 300, "replay_budget_bytes": 65536,
                                  "ewc_lambda": args.ewc_lambda, "fisher_samples": args.fisher_samples,
                                  "fisher_type": "per-example empirical diagonal; task-wise snapshots; skip last task",
                                  "der_alpha": args.der_alpha, "der_beta": args.der_beta,
                                  "der_loss": "CE(current)+alpha*mean(logit_error^2)+beta*CE(independent replay)",
                                  "selection": "Fixed defaults, not tuned using final test set"},
              "mnema_parameters": MNEMA_PARAMETERS, "mnema_initial_settings": mnema_defaults(),
              "source_sha256": source_manifest(),
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__,
                           "platform": platform.platform(), "processor": platform.processor(), "blas_threads": 1},
              "protocol": {"class_il": True, "output_classes": list(range(10)), "task_id_at_inference": False,
                           "evaluation": "Each image from identical trained checkpoint; all five tasks at each checkpoint",
                           "forgetting": "mean_j=0..3(max_t=j..3 R[t,j] - R[4,j]); negative allowed",
                           "seed_stream": "default_rng(SeedSequence([seed,101])); generated once per seed",
                           "quick_selection": "first N of shuffled training stream; fixed random test subset seed 2026",
                           "memory": "Unique resident NumPy array payload; excludes Python objects, PRNG internals, dataset and scratch. Buffer counters included.",
                           "faststore_monitor": "after every external training sample; peaks do not include transient within-step occupancy"}}
    config["config_id"] = digest(config)
    config["created_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        config["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        config["git_commit"] = None
    return config


def main(argv=None):
    args = parse_args(argv)
    data = load_dataset(ROOT / "data")
    config = build_config(args, data)
    output = ROOT / "results/research_benchmark" / args.mode
    if args.seed is not None:
        method = selected_method(args.method)
        trains, tests, _ = task_indices(data, args.seed, args.mode == "quick",
                                        args.quick_train, args.quick_test)
        pair_output = output / "pairs" / f"seed_{args.seed}_{args.method}"
        atomic_json(pair_output / "config.json", config)
        atomic_json(pair_output / "raw" / f"seed_{args.seed}_{args.method}.json",
                    run_one(data, trains, tests, args.seed, method, config))
        atomic_json(pair_output / "status.json", {"status": "complete", "mode": args.mode,
                                                   "seed": args.seed, "method": method,
                                                   "config_id": config["config_id"]})
        print(f"{config['label']}: finished pair {args.seed} | {method}. Outputs: {pair_output}", flush=True)
        return
    config_path = output / "config.json"
    if args.force and output.exists():
        archive = output.parent / (args.mode + "_archive_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        # Both fully resolved paths must remain inside the research output directory.
        if output.resolve().parent != (ROOT / "results/research_benchmark").resolve() or archive.resolve().parent != output.resolve().parent:
            raise ValueError("Unsafe archive path")
        output.rename(archive)
        print(f"Archived previous outputs to {archive}")
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["config_id"] != config["config_id"]:
            raise ValueError("Existing configuration/source/environment differs; use --force to archive and rerun")
        config = previous
    atomic_json(config_path, config)
    results = []
    atomic_json(output / "status.json", {"status": "running", "config_id": config["config_id"]})
    for seed in config["seeds"]:
        trains, tests, _ = task_indices(data, seed, args.mode == "quick", args.quick_train, args.quick_test)
        stream_hashes = [sequence_hash(i) for i in trains]
        atomic_json(output / "raw" / f"seed_{seed}_indices.json",
                    {"config_id": config["config_id"], "train": [i.tolist() for i in trains], "test": [i.tolist() for i in tests]})
        for method, slug in METHODS.items():
            path = output / "raw" / f"seed_{seed}_{slug}.json"
            result = None
            if path.exists():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if valid_result(cached, config["config_id"], seed, method, stream_hashes):
                        result = cached
                        print(f"Resume: verified and skipped seed {seed} | {method}", flush=True)
                except (ValueError, OSError):
                    pass
            if result is None:
                result = run_one(data, trains, tests, seed, method, config)
                atomic_json(path, result)
            results.append(result)
            atomic_json(output / "summaries/summary.json", aggregate(config, results))
    finish_run(output, config, results)


def finish_run(output, config, results):
    """Write the standard report bundle after every expected pair is available."""
    atomic_json(output / "raw/per_seed_results.json", results)
    atomic_json(output / "full_results.json", {"config": config, "runs": results})
    summary = aggregate(config, results)
    atomic_json(output / "summaries/summary.json", summary)
    from experiments.plot_research_benchmark import generate_plots
    from experiments.report_research_benchmark import generate_report
    generate_plots(output)
    generate_report(output)
    atomic_json(output / "status.json", {"status": "complete", "mode": config["mode"],
                                       "research_complete": summary["research_complete"], "config_id": config["config_id"]})
    print(f"{config['label']}: finished. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()

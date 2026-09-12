# Python API

[Documentation index](README.md) · [Architecture](ARCHITECTURE.md)

Run these examples from the repository root with the [research dependencies](GETTING_STARTED.md). The implementation is imported from the checkout; the packaged console entry point is a placeholder. These are prototype interfaces, without a declared stable API or model-checkpoint serialization format.

## Train and predict one example

This synthetic input checks integration without downloading MNIST. It does not demonstrate learned accuracy.

```python
import numpy as np

from instrument.counters import EnergyInstrument
from mnema.model import MNEMA

np.random.seed(0)  # Seed before construction: connectivity and weights use NumPy RNG.
model = MNEMA(n_in=784, n_s=16384, k=64, d=10, budget_bytes=65536)
instrument = EnergyInstrument("instrument/tech/asic_45nm.yaml")
x = np.linspace(0.0, 1.0, 784, dtype=np.float32)

with instrument:
    training = model.step(x, y=3, is_training=True, instrument=instrument)

print("Prediction before this update:", training["prediction"])
assert training["probabilities"].shape == (10,)
assert np.isclose(training["probabilities"].sum(), 1.0)

instrument.counts.reset()  # Counters otherwise accumulate across calls.
with instrument:
    inference = model.step(x, is_training=False, instrument=instrument)

projection = instrument.project_energy()
print("Native prediction:", inference["prediction"])
print("Projected microjoules:", projection["energy_microjoules"])
assert projection["is_measured"] is False
```

`is_training=False` still changes adaptive inference state. For independent held-out predictions, use the benchmark adapter described below.

## MNEMA constructor

Source: [mnema/model.py](../mnema/model.py).

| Argument | Default | Meaning |
| --- | ---: | --- |
| `n_in` | 784 | Number of flattened input values |
| `n_s` | 16384 | Separator expansion and FastStore row count |
| `k` | 64 | Active separator indices per input |
| `d` | 10 | Output classes, indexed from zero |
| `budget_bytes` | 65536 | Configured FastStore occupancy threshold; actual row payload is undercounted |

These are model arguments, not benchmark tuning recommendations. Research uses the fixed defaults. Custom dimensions must satisfy the implementation's array operations: `n_in >= 8` for fan-in sampling, a positive `k` no larger than the active separator pool, and a positive class count. Comprehensive argument validation is not implemented.

## step

```python
model.step(x, y=None, is_training=True, instrument=None)
```

| Input | Contract |
| --- | --- |
| `x` | Finite NumPy intensity array whose flattened size equals `n_in`; float32 values in `[0, 1]` match the research preprocessing |
| `y` | Integer class in `[0, d)` for supervised learning; omit for inference |
| `is_training` | Controls eligibility maintenance and, with a label, learning and consolidation |
| `instrument` | Optional `EnergyInstrument` to accumulate selected operation counts |

Calling with `y=None` and the default `is_training=True` maintains eligibility but does not learn. For prediction-only use, explicitly set `is_training=False`. The method does not perform complete shape, range, or finite-value validation for you.

The returned dictionary contains:

| Key | Value |
| --- | --- |
| `prediction` | Integer argmax of the blended probabilities |
| `probabilities` | Length-`d` NumPy softmax array |
| `fast_weight_alpha` | Scalar gate weighting the FastStore contribution |
| `confidence` | FastStore vote-margin score; not a calibrated probability of correctness |
| `modulator_bus` | Mutable controller object with `nu`, `delta`, `sigma`, `phi` |

The bus is a live object reused by the model. Copy values with `vars(output["modulator_bus"]).copy()` when retaining telemetry across steps. Predictions and probabilities describe the forward pass before any subsequent learning in that call.

## Checkpoint-isolated prediction

The experiment adapter is specialized to the ten-class benchmark. It accepts a raw uint8 image and normalizes it by 255. Do not pass an already normalized float image to this example.

```python
import numpy as np

from experiments.research_state import CheckpointInference, state_hash
from mnema.model import MNEMA

np.random.seed(0)
model = MNEMA()
raw_image = np.arange(784, dtype=np.uint16).astype(np.uint8)
before = state_hash(model)
checkpoint = CheckpointInference(model)

first = checkpoint.predict(raw_image)
second = checkpoint.predict(raw_image)
assert first == second
assert state_hash(model) == before
```

Create the adapter after training a checkpoint and recreate it after further training. Its `predict` method restores the captured state after each call, including after an exception. Do not interleave unrelated model updates with a retained adapter. For dataset evaluation with integrity and reverse-order checks, use [evaluate](../experiments/research_state.py) through the benchmark runner.

## EnergyInstrument

```python
from instrument.counters import EnergyInstrument

instrument = EnergyInstrument("instrument/tech/asic_45nm.yaml")
instrument.counts.reset()
with instrument:
    instrument.macs += 100
projection = instrument.project_energy()
assert projection["is_measured"] is False
```

The instrument proxies fields such as `macs`, `adds`, `synops`, `neuron_updates`, SRAM/DRAM bytes, `spikes`, and `weight_updates` to its `counts` dataclass. Reading `vars(instrument.counts).copy()` gives a snapshot.

Operation counters accumulate until `counts.reset()`. Each context-manager exit **replaces** `wall_time_s` with the duration of that context; it does not accumulate elapsed time across contexts. For one coherent timed interval, use one context around the desired workload and reset counters beforehand. Avoid nesting the same instrument.

`project_energy()` returns joules, microjoules, `edp`, technology source, a disclaimer, and `is_measured`. `edp` multiplies the projection by the stored wall time. Read the [accounting limits](ARCHITECTURE.md#operation-counts-and-energy) before comparing projections.

## Component access and persistence

The model exposes `encoder`, `separator`, `store`, `cortex`, `readout`, `controller`, and `consolidator`. Direct access is useful for diagnostics, but can invalidate benchmark assumptions. For example, `cortex.reset_state()` clears neuron and eligibility state; it is not a complete model reset.

The runner saves results and indices, not resumable model weights during training. There is no supported `save()` / `load()` API. [Resume](../experiments/RESEARCH_BENCHMARK.md#resume-and-fresh-runs) reuses complete seed-method results and restarts interrupted pairs.

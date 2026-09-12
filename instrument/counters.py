# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import time
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os

@dataclass
class OpCounts:
    synops: int = 0             # Spikes arriving at a non-zero synapse
    macs: int = 0               # Multiply-accumulate operations (dense layers)
    adds: int = 0               # Pure integer/float additions (sparse store)
    neuron_updates: int = 0     # LIF/ALIF state updates
    sram_read_b: int = 0        # Bytes read from SRAM
    sram_write_b: int = 0       # Bytes written to SRAM
    dram_read_b: int = 0        # Bytes read from DRAM
    dram_write_b: int = 0       # Bytes written to DRAM
    spikes: int = 0             # Total events emitted
    weight_updates: int = 0     # Synaptic weights modified
    wall_time_s: float = 0.0

    def reset(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, 0 if field_name != 'wall_time_s' else 0.0)

class EnergyInstrument:
    def __init__(self, tech_card_path: str = "instrument/tech/asic_45nm.yaml"):
        self.counts = OpCounts()
        self.tech_card = self._load_tech_card(tech_card_path)
        self._start_time = 0.0

    def _load_tech_card(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Technology card not found at {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def __getattr__(self, item):
        if hasattr(self.counts, item):
            return getattr(self.counts, item)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in ("counts", "tech_card", "_start_time"):
            super().__setattr__(key, value)
        elif hasattr(self, "counts") and hasattr(self.counts, key):
            setattr(self.counts, key, value)
        else:
            super().__setattr__(key, value)

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.counts.wall_time_s = time.perf_counter() - self._start_time

    def project_energy(self) -> Dict[str, float]:
        """Calculates projected energy (Joules) based on cited technology card specs."""
        energy_j = 0.0
        
        if "synop_J" in self.tech_card:
            energy_j += self.counts.synops * self.tech_card["synop_J"]
        if "neuron_update_J" in self.tech_card:
            energy_j += self.counts.neuron_updates * self.tech_card["neuron_update_J"]
        if "mac_fp32_J" in self.tech_card:
            energy_j += self.counts.macs * self.tech_card["mac_fp32_J"]
        if "add_int32_J" in self.tech_card:
            energy_j += self.counts.adds * self.tech_card["add_int32_J"]
        if "sram_read_32b_J" in self.tech_card:
            reads_32b = self.counts.sram_read_b / 4.0
            energy_j += reads_32b * self.tech_card["sram_read_32b_J"]
        if "dram_read_32b_J" in self.tech_card:
            dram_reads_32b = self.counts.dram_read_b / 4.0
            energy_j += dram_reads_32b * self.tech_card["dram_read_32b_J"]

        edp = energy_j * self.counts.wall_time_s

        return {
            "energy_joules": energy_j,
            "energy_microjoules": energy_j * 1e6,
            "edp": edp,
            "tech_source": self.tech_card.get("source", "Unknown"),
            "is_measured": False,
            "disclaimer": "PROJECTED: Measured hardware op-counts multiplied by published technology card."
        }
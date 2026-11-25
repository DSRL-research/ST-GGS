"""Hourly AC power-flow labels produced by `stggs.power_flow.generate_labels`."""
import glob
import json
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def _metadata_path(labels_dir: str) -> str:
    """The label metadata file: `pf_meta.json` or a prefixed `*_pf_meta.json`."""
    default = os.path.join(labels_dir, "pf_meta.json")
    if os.path.exists(default):
        return default
    matches = sorted(glob.glob(os.path.join(labels_dir, "*pf_meta.json")))
    if not matches:
        raise FileNotFoundError(f"no pf_meta.json in {labels_dir}")
    return matches[0]


class PowerFlowLabels:
    """Solved state and injections per hour, indexed in the topology's substation order.

    `row_of` maps the timestamp of each converged hour to its row.
    """

    def __init__(self, labels_dir: str, sub_name: np.ndarray):
        with open(_metadata_path(labels_dir)) as f:
            meta = json.load(f)
        bus = np.load(os.path.join(labels_dir, "pf_bus.npz"))

        column_of_bus = {int(b): i for i, b in enumerate(meta["bus_ids"])}
        columns = np.array([column_of_bus[int(name)] for name in sub_name])
        self.vm = bus["vm_pu"][:, columns].astype(np.float32)
        self.va = bus["va_degree"][:, columns].astype(np.float32)
        self.p_inj = bus["p_inj"][:, columns].astype(np.float32)
        self.q_inj = bus["q_inj"][:, columns].astype(np.float32)
        self.q_shunt = bus["q_shunt"][:, columns].astype(np.float32)

        converged = np.array(meta["converged"], dtype=bool)
        finite = (np.isfinite(self.vm).all(axis=1) & np.isfinite(self.va).all(axis=1)
                  & np.isfinite(self.p_inj).all(axis=1) & np.isfinite(self.q_inj).all(axis=1))
        usable = converged & finite
        timestamps = pd.to_datetime(meta["timestamps"])
        self.row_of: Dict[pd.Timestamp, int] = {t: i for i, (t, u) in enumerate(zip(timestamps, usable)) if u}

    def lookup(self, hist_ts: np.ndarray) -> int:
        """Row of the last hour of a lookback, or -1 when that hour did not converge."""
        return self.row_of.get(pd.Timestamp(hist_ts[-1]), -1)

    def target_scales(self, rows: Sequence[int], f_idx: np.ndarray, t_idx: np.ndarray) -> Tuple[float, float]:
        """(std of Vm, std of per-branch angle difference in radians) over `rows`."""
        rows = np.asarray([r for r in rows if r >= 0], dtype=np.int64)
        vm_scale = float(self.vm[rows].std())
        angles = np.deg2rad(self.va[rows])
        difference = wrap_angle(angles[:, f_idx] - angles[:, t_idx])
        return (vm_scale if vm_scale > 1e-9 else 1.0), (float(difference.std()) or 1.0)


def wrap_angle(radians: np.ndarray) -> np.ndarray:
    """Wraps angles into [-pi, pi)."""
    return (radians + np.pi) % (2 * np.pi) - np.pi

"""The pandapower network, the hourly series aggregated to buses, and the AC solver."""
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_UNIT_COLUMN = re.compile(r"^(?:Bus|Gen)\s+(\d+)\b")
_BARE_BUS_COLUMN = re.compile(r"^(\d+)(?:\.\d+)?$")


@dataclass
class Network:
    net: object
    case: object
    bus_ids: np.ndarray
    bus_position: Dict[int, int]
    branch_row: np.ndarray
    branch_table: np.ndarray
    branch_flipped: np.ndarray


def build_network(case_file: str) -> Network:
    """Builds the pandapower net and maps its elements back to the case's branch order."""
    from matpowercaseframes import CaseFrames
    from pandapower.converter.pypower import from_ppc

    case = CaseFrames(case_file)
    ppc = {"version": "2", "baseMVA": 100.0, "bus": case.bus.to_numpy(dtype=float),
           "gen": case.gen.to_numpy(dtype=float), "branch": case.branch.to_numpy(dtype=float)}
    net = from_ppc(ppc, f_hz=60, validate_conversion=False)
    bus_ids = case.bus["BUS_I"].astype(int).to_numpy()

    pending: Dict[Tuple[int, int], list] = {}
    from_bus = case.branch["F_BUS"].astype(int).to_numpy()
    to_bus = case.branch["T_BUS"].astype(int).to_numpy()
    for i, endpoints in enumerate(zip(from_bus, to_bus)):
        pending.setdefault(endpoints, []).append(i)

    row = np.full(len(case.branch), -1, dtype=np.int64)
    table_of = np.full(len(case.branch), -1, dtype=np.int8)
    flipped = np.zeros(len(case.branch), dtype=bool)
    tables = ((0, net.line, "from_bus", "to_bus"), (1, net.impedance, "from_bus", "to_bus"),
              (2, net.trafo, "hv_bus", "lv_bus"))
    for table_id, table, from_column, to_column in tables:
        for table_row, (f, t) in enumerate(zip(table[from_column].to_numpy(), table[to_column].to_numpy())):
            key, flip = (int(f), int(t)), False
            if not pending.get(key):
                key, flip = (int(t), int(f)), True
            if not pending.get(key):
                continue
            i = pending[key].pop(0)
            row[i], table_of[i], flipped[i] = table_row, table_id, flip
    if (row < 0).any():
        raise RuntimeError(f"{int((row < 0).sum())} case branches have no pandapower element")

    return Network(net, case, bus_ids, {b: i for i, b in enumerate(bus_ids)}, row, table_of, flipped)


def branch_results(network: Network) -> np.ndarray:
    """(4, n_branch) p_from, q_from, p_to, q_to in the case's branch order."""
    net = network.net
    row, table_of, flipped = network.branch_row, network.branch_table, network.branch_flipped
    out = np.empty((4, len(row)), dtype=np.float64)
    results = ((0, net.res_line, "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"),
               (1, net.res_impedance, "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"),
               (2, net.res_trafo, "p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar"))
    for table_id, table, p_f, q_f, p_t, q_t in results:
        selected = table_of == table_id
        if not selected.any():
            continue
        rows = row[selected]
        pf, qf = table[p_f].to_numpy()[rows], table[q_f].to_numpy()[rows]
        pt, qt = table[p_t].to_numpy()[rows], table[q_t].to_numpy()[rows]
        flip = flipped[selected]
        out[0, selected] = np.where(flip, pt, pf)
        out[1, selected] = np.where(flip, qt, qf)
        out[2, selected] = np.where(flip, pf, pt)
        out[3, selected] = np.where(flip, qf, qt)
    return out


def _series_per_bus(path: str) -> pd.DataFrame:
    """One yearly CSV aggregated to one column per bus, indexed by timestamp."""
    raw = pd.read_csv(path, header=1, low_memory=False)
    pairs = []
    for column in raw.columns:
        match = _UNIT_COLUMN.match(str(column)) or _BARE_BUS_COLUMN.match(str(column))
        if match is not None:
            pairs.append((column, int(match.group(1))))
    columns, buses = zip(*pairs)
    values = raw[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values.columns = list(buses)
    per_bus = values.T.groupby(level=0).sum().T
    per_bus.index = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), format="mixed")
    return per_bus


def read_bus_series(load_mw_csv: str, load_mvar_csv: str,
                    renewable_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-bus hourly demand P, demand Q and renewable P, joined on common timestamps."""
    load_p, load_q, renewable = (_series_per_bus(p) for p in (load_mw_csv, load_mvar_csv, renewable_csv))
    common = load_p.index.intersection(load_q.index).intersection(renewable.index).sort_values()
    return load_p.loc[common], load_q.loc[common], renewable.loc[common]


class Solver:
    """Writes a bus-level operating point onto the net and solves the AC power flow."""

    _INITIALISATIONS = (dict(init="dc"), dict(init="flat"), dict(init="dc", max_iteration=100))

    def __init__(self, network: Network):
        self.network = network
        net = network.net
        self.n_bus = len(network.bus_ids)
        position = np.vectorize(network.bus_position.get)
        self.load_position = position(net.load.bus.to_numpy())
        self.gen_position = position(net.gen.bus.to_numpy())
        self.sgen_position = position(net.sgen.bus.to_numpy()) if len(net.sgen) else np.array([], dtype=int)
        # Generation at a bus holding both element kinds is written to the gen row only.
        self.sgen_shares_bus = (np.isin(self.sgen_position, self.gen_position)
                                if len(self.sgen_position) else np.array([], dtype=bool))

    def solve(self, load_p: np.ndarray, load_q: np.ndarray,
              gen_p: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Solves with distributed slack. Returns (converged, Vm, Va in degrees) in case bus order."""
        import pandapower as pp

        net = self.network.net
        net.load.loc[:, "p_mw"] = load_p[self.load_position]
        net.load.loc[:, "q_mvar"] = load_q[self.load_position]
        net.gen.loc[:, "p_mw"] = gen_p[self.gen_position]
        if len(net.sgen):
            net.sgen.loc[:, "p_mw"] = np.where(self.sgen_shares_bus, 0.0, gen_p[self.sgen_position])
        net.gen.loc[:, "slack_weight"] = 1.0

        for options in self._INITIALISATIONS:
            try:
                pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True, numba=False, **options)
            except Exception:
                continue
            if bool(net.converged):
                result = net.res_bus.loc[self.network.bus_ids]
                return True, result.vm_pu.to_numpy(), result.va_degree.to_numpy()
        return False, None, None


def require_files(*paths: str) -> None:
    """Raises when any of `paths` does not exist."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing input file(s):\n  " + "\n  ".join(missing))

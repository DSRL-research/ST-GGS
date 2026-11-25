"""Loader for the ACTIVSg synthetic grids.

Reads the MATPOWER case, the PowerWorld .aux file and the yearly hourly CSVs and
returns the grid `Topology` plus one tidy frame with a column per node and channel:
`load_p_*`, `load_q_*`, `prod_p_*` and `prod_v_*`. The generator second channel
holds the capacity factor (P / Pmax). Generators are the renewable units, which
are the only ones with published time series.

The year is cut into pseudo-episodes of consecutive ISO weeks, named
`<iso_year>_block_<nn>`, so that block names sort chronologically.
"""
import re
import shlex
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from stggs.config import Config

_UNIT_COLUMN = re.compile(r"(?:Bus|Gen)\s+(\d+)\s+#(\d+)")
_BARE_BUS_COLUMN = re.compile(r"^(\d+)(?:\.\d+)?$")


@dataclass
class Topology:
    n_sub: int
    n_load: int
    n_gen: int
    n_line: int
    sub_name: np.ndarray
    load_name: np.ndarray
    gen_name: np.ndarray
    load_to_subid: np.ndarray
    gen_to_subid: np.ndarray
    line_or_to_subid: np.ndarray
    line_ex_to_subid: np.ndarray
    thermal_limit: np.ndarray
    line_r: np.ndarray
    line_x: np.ndarray
    line_b: np.ndarray
    line_tap: np.ndarray
    line_shift: np.ndarray
    sub_base_kv: np.ndarray
    sub_bus_type: np.ndarray
    sub_gen_vg: np.ndarray
    sub_shunt_gs: np.ndarray
    sub_shunt_bs: np.ndarray
    sub_info: np.ndarray
    sub_coords: np.ndarray
    gen_pmin: np.ndarray
    gen_pmax: np.ndarray
    gen_ramp: np.ndarray
    gen_type: np.ndarray
    load_static: np.ndarray


def _parse_aux_block(aux_path: str, table: str) -> Tuple[List[str], List[List[str]]]:
    """Column names and raw rows of a `DATA (<table>, [...])` block of an .aux file."""
    with open(aux_path) as f:
        lines = f.readlines()

    start = next((i for i, l in enumerate(lines) if l.startswith(f"DATA ({table},")), None)
    if start is None:
        raise ValueError(f"no 'DATA ({table}, [...])' block in {aux_path}")
    header_end = next(i for i in range(start, len(lines)) if "])" in lines[i])
    header = "".join(lines[start:header_end + 1])
    columns = [c.strip() for c in header.split("[", 1)[1].split("])", 1)[0].replace("\n", " ").split(",")]
    open_brace = next(i for i in range(header_end, len(lines)) if lines[i].strip() == "{")
    close_brace = next(i for i in range(open_brace, len(lines)) if lines[i].strip() == "}")
    rows = [shlex.split(l, posix=True) for l in lines[open_brace + 1:close_brace] if l.strip()]
    return columns, rows


def _bus_coordinates(aux_path: str) -> Dict[int, Tuple[float, float]]:
    """{bus number: (latitude, longitude)} from the .aux Bus block."""
    cols, rows = _parse_aux_block(aux_path, "Bus")
    bus, lat, lon = cols.index("BusNum"), cols.index("Latitude:1"), cols.index("Longitude:1")
    return {int(r[bus]): (float(r[lat]), float(r[lon])) for r in rows}


def _load_statics(aux_path: str, load_names, bus_base_kv: Dict[int, float]) -> np.ndarray:
    """(n_load, 3) nominal MW, nominal MVAr and bus base kV, in physical units."""
    cols, rows = _parse_aux_block(aux_path, "Load")
    bus, load_id = cols.index("BusNum"), cols.index("LoadID")
    mw, mvar = cols.index("LoadSMW"), cols.index("LoadSMVR")
    by_name = {}
    for r in rows:
        b = int(r[bus])
        by_name[f"{b}_{r[load_id].strip()}"] = (float(r[mw]), float(r[mvar]), bus_base_kv.get(b, 0.0))
    return np.array([by_name[n] for n in load_names], dtype=np.float32)


def _name_bare_columns(df: pd.DataFrame, columns: List[str], types: Dict[str, str],
                       aux_path: str) -> Dict[str, str]:
    """Names the columns whose header is only a bus number.

    Each bus's columns are matched to its .aux generators of the same fuel by
    pairing the largest observed output with the largest nameplate capacity.
    """
    cols, rows = _parse_aux_block(aux_path, "Gen")
    bus_i, id_i = cols.index("BusNum"), cols.index("GenID")
    fuel_i, mw_i = cols.index("GenFuelType"), cols.index("GenMWMax")
    units_at_bus: Dict[int, List[Tuple[str, str, float]]] = {}
    for r in rows:
        units_at_bus.setdefault(int(r[bus_i]), []).append(
            (r[id_i], r[fuel_i].strip().lower(), float(r[mw_i])))

    by_bus: Dict[str, List[str]] = {}
    for c in columns:
        by_bus.setdefault(_BARE_BUS_COLUMN.match(c).group(1), []).append(c)

    names: Dict[str, str] = {}
    for bus, group in by_bus.items():
        fuel = types[group[0]].lower()
        candidates = [(gid, cap) for gid, f, cap in units_at_bus.get(int(bus), []) if f == fuel]
        if len(candidates) != len(group):
            names.update({c: f"{bus}_{i}" for i, c in enumerate(group, start=1)})
            continue
        observed_max = df[group].max(axis=0)
        ordered_columns = sorted(group, key=lambda c: observed_max[c])
        ordered_units = sorted(candidates, key=lambda u: u[1])
        names.update({c: f"{bus}_{gid}" for c, (gid, _) in zip(ordered_columns, ordered_units)})
    return names


def _read_series(path: str, aux_path: str) -> Tuple[pd.DataFrame, list]:
    """One yearly CSV as (frame indexed by timestamp, per-column type labels).

    Columns are renamed to `<bus>_<unit>`.
    """
    type_row = pd.read_csv(path, nrows=0).columns.tolist()
    df = pd.read_csv(path, header=1)
    timestamps = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %I:%M:%S %p")

    value_columns = [c for c in df.columns if _UNIT_COLUMN.search(c) or _BARE_BUS_COLUMN.match(c)]
    out = df[value_columns].copy()
    out.index = pd.DatetimeIndex(timestamps, name="timestamp")
    types = [t.split(".")[0] for t in type_row[5:5 + len(value_columns)]]
    column_types = dict(zip(value_columns, types))

    names: Dict[str, str] = {}
    for c in value_columns:
        match = _UNIT_COLUMN.search(c)
        if match:
            names[c] = f"{match.group(1)}_{match.group(2)}"
    bare = [c for c in value_columns if c not in names]
    if bare:
        names.update(_name_bare_columns(df, bare, column_types, aux_path))

    out.columns = [names[c] for c in value_columns]
    return out, types


def _generator_setpoints(case, bus_ids) -> np.ndarray:
    """Generator voltage setpoint per bus, 0.0 where the bus has no generator."""
    position = {b: i for i, b in enumerate(bus_ids)}
    setpoints = np.zeros(len(bus_ids), dtype=np.float32)
    for bus, vg in zip(case.gen["GEN_BUS"].astype(int), case.gen["VG"].astype(float)):
        i = position.get(int(bus))
        if i is not None:
            setpoints[i] = max(setpoints[i], float(vg))
    return setpoints


def _generator_limits(case, gen_names, renewable: pd.DataFrame):
    """(Pmax, Pmin, ramp) per renewable unit.

    Taken from the case when exactly one case generator sits at the unit's bus;
    otherwise Pmax is the unit's own yearly maximum and Pmin and ramp are zero.
    """
    gens = case.gen
    ramp_column = "RAMP_10" if "RAMP_10" in gens.columns else None
    pmax, pmin, ramp = [], [], []
    for name in gen_names:
        bus = int(name.split("_")[0])
        rows = gens[gens["GEN_BUS"].astype(int) == bus]
        if len(rows) == 1:
            pmax.append(float(rows["PMAX"].iloc[0]))
            pmin.append(float(rows["PMIN"].iloc[0]))
            ramp.append(float(rows[ramp_column].iloc[0]) if ramp_column else 0.0)
        else:
            pmax.append(float(renewable[name].max()))
            pmin.append(0.0)
            ramp.append(0.0)
    return (np.array(pmax, dtype=np.float32), np.array(pmin, dtype=np.float32),
            np.array(ramp, dtype=np.float32))


def load_activsg(cfg: Config) -> Tuple[Topology, pd.DataFrame]:
    """Returns the grid topology and the hourly frame with an `episode_id` column."""
    from matpowercaseframes import CaseFrames

    case = CaseFrames(cfg.case_file)
    bus_ids = case.bus["BUS_I"].astype(int).tolist()
    bus_index = {b: i for i, b in enumerate(bus_ids)}

    load_p, _ = _read_series(cfg.load_mw_csv, cfg.aux_file)
    load_q, _ = _read_series(cfg.load_mvar_csv, cfg.aux_file)
    prod_p, gen_types = _read_series(cfg.renewable_csv, cfg.aux_file)
    if list(load_p.columns) != list(load_q.columns):
        raise ValueError("the load MW and MVAr files list different load points")

    common = load_p.index.intersection(load_q.index).intersection(prod_p.index)
    load_p, load_q, prod_p = (frame.loc[common] for frame in (load_p, load_q, prod_p))

    load_names = np.array(load_p.columns)
    gen_names = np.array(prod_p.columns)
    load_to_subid = np.array([bus_index[int(c.split("_")[0])] for c in load_names])
    gen_to_subid = np.array([bus_index[int(c.split("_")[0])] for c in gen_names])

    branch = case.branch
    line_or = np.array([bus_index[int(b)] for b in branch["F_BUS"]])
    line_ex = np.array([bus_index[int(b)] for b in branch["T_BUS"]])

    n_sub = len(bus_ids)
    sub_info = np.zeros(n_sub, dtype=np.float32)
    for endpoints in (load_to_subid, gen_to_subid, line_or, line_ex):
        np.add.at(sub_info, endpoints, 1.0)

    pmax, pmin, ramp = _generator_limits(case, gen_names, prod_p)
    base_kv = dict(zip(case.bus["BUS_I"].astype(int), case.bus["BASE_KV"].astype(float)))
    coordinates = _bus_coordinates(cfg.aux_file)

    topology = Topology(
        n_sub=n_sub,
        n_load=len(load_names),
        n_gen=len(gen_names),
        n_line=len(branch),
        sub_name=np.array([str(b) for b in bus_ids]),
        load_name=load_names,
        gen_name=gen_names,
        load_to_subid=load_to_subid,
        gen_to_subid=gen_to_subid,
        line_or_to_subid=line_or,
        line_ex_to_subid=line_ex,
        thermal_limit=branch["RATE_A"].to_numpy(dtype=np.float32),
        line_r=branch["BR_R"].to_numpy(dtype=np.float32),
        line_x=branch["BR_X"].to_numpy(dtype=np.float32),
        line_b=branch["BR_B"].to_numpy(dtype=np.float32),
        line_tap=branch["TAP"].to_numpy(dtype=np.float32),
        line_shift=branch["SHIFT"].to_numpy(dtype=np.float32),
        sub_base_kv=case.bus["BASE_KV"].to_numpy(dtype=np.float32),
        sub_bus_type=case.bus["BUS_TYPE"].to_numpy(dtype=np.float32),
        sub_gen_vg=_generator_setpoints(case, bus_ids),
        sub_shunt_gs=case.bus["GS"].to_numpy(dtype=np.float32),
        sub_shunt_bs=case.bus["BS"].to_numpy(dtype=np.float32),
        sub_info=sub_info,
        sub_coords=np.array([coordinates[b] for b in bus_ids], dtype=np.float32),
        gen_pmin=pmin,
        gen_pmax=pmax,
        gen_ramp=ramp,
        gen_type=np.array([t.lower() for t in gen_types]),
        load_static=_load_statics(cfg.aux_file, load_names, base_kv),
    )

    capacity_factor = {
        name: prod_p[name] / max(float(pmax[i]), 1e-6) for i, name in enumerate(gen_names)
    }
    frame = pd.DataFrame(
        {
            **{f"load_p_{c}": load_p[c] for c in load_p.columns},
            **{f"load_q_{c}": load_q[c] for c in load_q.columns},
            **{f"prod_p_{c}": prod_p[c] for c in prod_p.columns},
            **{f"prod_v_{c}": capacity_factor[c] for c in prod_p.columns},
        },
        index=common,
    )

    iso = frame.index.isocalendar()
    weeks_per_episode = max(1, -(-(cfg.lookback + cfg.horizon + 168) // 168))
    block = ((iso["week"] - 1) // weeks_per_episode).astype(int)
    frame["episode_id"] = [f"{y}_block_{b:02d}" for y, b in zip(iso["year"], block)]
    return topology, frame

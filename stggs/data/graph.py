"""Heterogeneous grid graph: static features, edges and per-window samples.

Node types are substations, loads and generators. Relations are the electrical
attachments, one symmetric substation-substation relation for lines (parallel
circuits merged) and geographic k-nearest-neighbour relations between peers of
the same type.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from stggs.config import Config
from stggs.data.activsg import Topology
from stggs.data.calendar_features import calendar_features
from stggs.data.labels import PowerFlowLabels, wrap_angle
from stggs.data.windows import Window

EdgeType = Tuple[str, str, str]
CategoryInfo = Dict[str, Tuple[np.ndarray, List[str], np.ndarray]]

_EARTH_RADIUS_KM = 6371.0
LINE_RELATION = ("substation", "connects", "substation")


def physical_log(x: np.ndarray) -> np.ndarray:
    """Sign-preserving log(1 + |x|), applied to quantities in absolute physical units."""
    return np.sign(x) * np.log1p(np.abs(x))


def _encode_statics(raw: np.ndarray) -> np.ndarray:
    return physical_log(raw.astype(np.float32)).astype(np.float32)


def _substation_statics(topology: Topology) -> np.ndarray:
    """(n_sub, 7): degree, PV flag, slack flag, has-setpoint flag, setpoint, shunt G, shunt B."""
    continuous = _encode_statics(np.column_stack([
        topology.sub_info, topology.sub_gen_vg, topology.sub_shunt_gs, topology.sub_shunt_bs]))
    flags = np.column_stack([
        (topology.sub_bus_type == 2).astype(np.float32),
        (topology.sub_bus_type == 3).astype(np.float32),
        (topology.sub_gen_vg > 0).astype(np.float32),
    ])
    return np.concatenate([continuous[:, :1], flags, continuous[:, 1:]], axis=1).astype(np.float32)


def _generator_specification(topology: Topology) -> np.ndarray:
    """(n_gen, 6): Pmin, Pmax, ramp up, ramp down, cost and redispatchable.

    Cost and redispatchable are zero for the renewable units of ACTIVSg.
    """
    n_gen = topology.n_gen
    return np.stack([topology.gen_pmin, topology.gen_pmax, topology.gen_ramp, topology.gen_ramp,
                     np.zeros(n_gen, dtype=np.float32), np.zeros(n_gen, dtype=np.float32)], axis=1)


def build_node_categories(topology: Topology, cfg: Config, load_magnitude: np.ndarray) -> CategoryInfo:
    """{node type: (category index per node, category names, specification vectors)}.

    Generators are grouped by fuel and loads by absolute size in MW. These groups
    feed the static contrastive objective.
    """
    fuels = sorted(set(topology.gen_type.tolist()))
    fuel_index = {fuel: i for i, fuel in enumerate(fuels)}
    generator_categories = np.array([fuel_index[t] for t in topology.gen_type], dtype=np.int64)

    edges = list(cfg.load_bucket_edges_mw)
    load_categories = np.digitize(load_magnitude, edges).astype(np.int64)
    bounds = [0.0] + edges + [float("inf")]
    bucket_names = [f"mw_{bounds[i]:g}_{bounds[i + 1]:g}" for i in range(len(bounds) - 1)]

    return {
        "load": (load_categories, bucket_names, _encode_statics(topology.load_static)),
        "generator": (generator_categories, fuels, _encode_statics(_generator_specification(topology))),
    }


def add_substation_aggregates(df: pd.DataFrame, topology: Topology) -> pd.DataFrame:
    """Adds `subp_*` and `subg_*`: total attached load and generation per substation, in MW."""
    load_columns = np.array([f"load_p_{n}" for n in topology.load_name])
    gen_columns = np.array([f"prod_p_{n}" for n in topology.gen_name])
    totals = {}
    for i, sub in enumerate(topology.sub_name):
        attached = load_columns[topology.load_to_subid == i]
        totals[f"subp_{sub}"] = df[list(attached)].sum(axis=1) if len(attached) else 0.0
        attached = gen_columns[topology.gen_to_subid == i]
        totals[f"subg_{sub}"] = df[list(attached)].sum(axis=1) if len(attached) else 0.0
    return pd.concat([df, pd.DataFrame(totals, index=df.index)], axis=1)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _geographic_knn_edges(coords: np.ndarray, k: int, chunk: int = 512) -> torch.Tensor:
    """Symmetric (2, E) edge index joining each node to its k nearest peers."""
    n = coords.shape[0]
    k = min(k, n - 1)
    lat, lon = coords[:, 0], coords[:, 1]
    sources, targets = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        distance = _haversine_km(lat[start:stop, None], lon[start:stop, None], lat[None, :], lon[None, :])
        distance[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
        sources.append(np.repeat(np.arange(start, stop), k))
        targets.append(nearest.reshape(-1))
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    both = np.concatenate([np.stack([sources, targets]), np.stack([targets, sources])], axis=1)
    return torch.tensor(np.unique(both, axis=1), dtype=torch.long)


def _build_edges(topology: Topology, k: int) -> Dict[EdgeType, torch.Tensor]:
    load_edges = torch.stack([torch.arange(topology.n_load, dtype=torch.long),
                              torch.tensor(topology.load_to_subid, dtype=torch.long)])
    gen_edges = torch.stack([torch.arange(topology.n_gen, dtype=torch.long),
                             torch.tensor(topology.gen_to_subid, dtype=torch.long)])
    lines = np.stack([topology.line_or_to_subid, topology.line_ex_to_subid])
    line_edges = torch.tensor(np.unique(np.concatenate([lines, lines[::-1]], axis=1), axis=1),
                              dtype=torch.long)
    return {
        ("load", "attached_to", "substation"): load_edges,
        ("substation", "rev_attached_to_load", "load"): load_edges.flip(0),
        ("generator", "attached_to", "substation"): gen_edges,
        ("substation", "rev_attached_to_gen", "generator"): gen_edges.flip(0),
        LINE_RELATION: line_edges,
        ("load", "near", "load"): _geographic_knn_edges(topology.sub_coords[topology.load_to_subid], k),
        ("generator", "near", "generator"): _geographic_knn_edges(topology.sub_coords[topology.gen_to_subid], k),
    }


def _corridor_parameters(indices: List[int], r, x, b, rating, tap):
    """Equivalent (R, X, B, rating, tap) of the parallel circuits of one corridor."""
    impedance = r[indices] + 1j * x[indices]
    impedance = np.where(np.abs(impedance) < 1e-9, 1e-9 + 0j, impedance)
    equivalent = 1.0 / np.sum(1.0 / impedance)
    return equivalent.real, equivalent.imag, b[indices].sum(), rating[indices].sum(), tap[indices].max()


def _build_edge_features(topology: Topology,
                         edges: Dict[EdgeType, torch.Tensor]) -> Dict[EdgeType, torch.Tensor]:
    """Per-edge attributes in absolute units.

    Lines: log-compressed equivalent R, X, B, admittance, rating and voltage step,
    plus the raw tap ratio. Peers: log distance in km. Attachments: the node's
    share of its substation's capacity and the log base voltage of the bus.
    """
    features: Dict[EdgeType, torch.Tensor] = {}

    corridors: Dict[tuple, list] = {}
    for i, (a, c) in enumerate(zip(topology.line_or_to_subid, topology.line_ex_to_subid)):
        corridors.setdefault((min(a, c), max(a, c)), []).append(i)

    r = np.asarray(topology.line_r, dtype=np.float64)
    x = np.asarray(topology.line_x, dtype=np.float64)
    b = np.asarray(topology.line_b, dtype=np.float64)
    rating = np.asarray(topology.thermal_limit, dtype=np.float64)
    tap = np.asarray(topology.line_tap, dtype=np.float64)
    kv = topology.sub_base_kv

    rows = []
    for source, target in zip(*edges[LINE_RELATION].numpy()):
        r_eq, x_eq, b_eq, rating_eq, tap_eq = _corridor_parameters(
            corridors[(min(source, target), max(source, target))], r, x, b, rating, tap)
        admittance = 1.0 / max(float(np.hypot(r_eq, x_eq)), 1e-9)
        voltage_step = abs(float(kv[source]) - float(kv[target]))
        compressed = physical_log(np.asarray(
            [r_eq, x_eq, b_eq, admittance, rating_eq, voltage_step], dtype=np.float32))
        rows.append(np.concatenate([compressed, np.asarray([tap_eq], dtype=np.float32)]).astype(np.float32))
    features[LINE_RELATION] = torch.tensor(np.stack(rows), dtype=torch.float32)

    magnitude = {"load": topology.load_static[:, 0], "generator": topology.gen_pmax}
    reverse = {"load": "rev_attached_to_load", "generator": "rev_attached_to_gen"}
    for node_type, mapping in (("load", topology.load_to_subid), ("generator", topology.gen_to_subid)):
        relation = (node_type, "near", node_type)
        peers = edges[relation].numpy()
        node_coords = topology.sub_coords[mapping]
        source, target = node_coords[peers[0]], node_coords[peers[1]]
        distance = _haversine_km(source[:, 0], source[:, 1], target[:, 0], target[:, 1])
        features[relation] = torch.tensor(physical_log(np.asarray(distance, dtype=np.float32))[:, None],
                                          dtype=torch.float32)

        size = np.abs(np.asarray(magnitude[node_type], dtype=np.float64))
        total = np.zeros(topology.n_sub, dtype=np.float64)
        np.add.at(total, mapping, size)
        share = size / np.maximum(total[mapping], 1e-6)
        attachment = np.stack([share, physical_log(np.asarray(kv[mapping], dtype=np.float32))], axis=1)
        attachment = torch.tensor(attachment.astype(np.float32), dtype=torch.float32)
        features[(node_type, "attached_to", "substation")] = attachment
        features[("substation", reverse[node_type], node_type)] = attachment
    return features


def build_graph_inputs(topology: Topology, cfg: Config):
    """Everything fixed across windows.

    Returns (static features, static widths, edge index, edge features).
    """
    statics = {
        "substation": _substation_statics(topology),
        "load": _encode_statics(topology.load_static),
        "generator": np.concatenate([
            _encode_statics(_generator_specification(topology)[:, :5]),
            np.zeros((topology.n_gen, 1), dtype=np.float32),
        ], axis=1),
    }
    widths = {node_type: values.shape[1] for node_type, values in statics.items()}
    edges = _build_edges(topology, cfg.geo_knn_k)
    return statics, widths, edges, _build_edge_features(topology, edges)


class WindowGraphDataset(torch.utils.data.Dataset):
    """Builds one `HeteroData` per window on access.

    With `labels`, each substation store also carries the state targets and
    injections of the window's last observed hour. P and Q at the slack bus and
    Q at PV buses are solver outputs, so they are zeroed in the input channels.
    """

    def __init__(self, windows: List[Window], topology: Topology, statics, edges, edge_features,
                 labels: Optional[PowerFlowLabels] = None):
        self.windows = windows
        self.topology = topology
        self.statics = statics
        self.edges = edges
        self.edge_features = edge_features
        self.labels = labels

        bus_type = np.asarray(topology.sub_bus_type)
        self.input_mask = np.ones((len(bus_type), 2), dtype=np.float32)
        self.input_mask[bus_type == 2, 1] = 0.0
        self.input_mask[bus_type == 3, :] = 0.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> HeteroData:
        window = self.windows[index]
        data = self._build_sample(window)
        if self.labels is not None:
            self._attach_state(data, self.labels.lookup(window.hist_ts))
        return data

    def _build_sample(self, window: Window) -> HeteroData:
        data = HeteroData()
        data["substation"].x = torch.from_numpy(np.concatenate(
            [window.sub_p_hist.T, window.sub_g_hist.T, self.statics["substation"]], axis=1)).float()
        data["load"].x = torch.from_numpy(np.concatenate(
            [window.load_p_hist.T, window.load_q_hist.T, self.statics["load"]], axis=1)).float()
        data["load"].y = torch.from_numpy(window.load_p_target.T).float()
        data["generator"].x = torch.from_numpy(np.concatenate(
            [window.prod_p_hist.T, window.prod_v_hist.T, self.statics["generator"]], axis=1)).float()
        data["generator"].y = torch.from_numpy(window.prod_p_target.T).float()

        for edge_type, edge_index in self.edges.items():
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_attr = self.edge_features[edge_type]

        data.cal_hist = torch.from_numpy(calendar_features(window.hist_ts)).float().unsqueeze(0)
        data.cal_future = torch.from_numpy(calendar_features(window.target_ts)).float().unsqueeze(0)
        return data

    def _attach_state(self, data: HeteroData, row: int) -> None:
        store = data["substation"]
        n_bus = self.topology.n_sub
        n_branch = self.topology.n_line
        if row < 0:
            store.y_vm = torch.zeros(n_bus, dtype=torch.float32)
            store.y_dtheta = torch.zeros(n_branch, dtype=torch.float32)
            store.pf_inj = torch.zeros(n_bus, 2, dtype=torch.float32)
            store.q_shunt = torch.zeros(n_bus, dtype=torch.float32)
            store.p_inj_mw = torch.zeros(n_bus, dtype=torch.float32)
            store.q_inj_mvar = torch.zeros(n_bus, dtype=torch.float32)
            store.state_valid = torch.zeros(1, dtype=torch.bool)
            return

        labels = self.labels
        angles = np.deg2rad(labels.va[row])
        difference = wrap_angle(angles[self.topology.line_or_to_subid] - angles[self.topology.line_ex_to_subid])
        injections = np.stack([labels.p_inj[row] / 100.0, labels.q_inj[row] / 100.0], axis=1)
        store.y_vm = torch.from_numpy(labels.vm[row]).float()
        store.y_dtheta = torch.from_numpy(difference).float()
        store.pf_inj = torch.from_numpy(injections * self.input_mask).float()
        store.q_shunt = torch.from_numpy(labels.q_shunt[row]).float()
        store.p_inj_mw = torch.from_numpy(labels.p_inj[row]).float()
        store.q_inj_mvar = torch.from_numpy(labels.q_inj[row]).float()
        store.state_valid = torch.ones(1, dtype=torch.bool)

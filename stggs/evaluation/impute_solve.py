"""Impute-solve pipeline (Table 2, Imp. -> PF columns).

Hidden loads are estimated, then passed with the true generation to the AC power
flow solver. Three estimators are compared: the jointly trained reconstruction
head, an imputation probe trained on the frozen backbone, and the per-bus training
mean. Reactive demand follows the active estimate at the bus's power factor.

    python -m stggs.evaluation.impute_solve --dataset activsg2000 --variant st_ggs
"""
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from stggs.config import add_experiment_arguments, config_from_args
from stggs.data.labels import PowerFlowLabels
from stggs.data.observability import hidden_node_mask, injection_channels, random_unobserved
from stggs.data.pipeline import Experiment, prepare_experiment, scaler_file
from stggs.evaluation.common import load_model, write_json
from stggs.models.st_ggs import STGGS
from stggs.power_flow.network import Solver, build_network, read_bus_series

HIDDEN_LEVELS = (0.5, 1.0)
METHODS = ("model", "model_probe", "per_bus_mean")
PROBE_STRIDE = 6
PROBE_EPOCHS = 60
PROBE_BATCH = 8192
CEILING_TOLERANCE_PU = 1e-3


def _tokens(model: STGGS, exp: Experiment, sample, observed: torch.Tensor, device):
    """(tokens, mask of loads hidden at the current hour) under the observability mask."""
    topology = exp.topology
    masks = hidden_node_mask(observed, topology.load_to_subid, topology.gen_to_subid, model.lookback_max,
                             topology.n_load, topology.n_gen, device=device)
    injections = injection_channels(sample["substation"].pf_inj, observed.to(device))
    tokens = model.tokens(*model.encode(sample, mask_dict=masks, injections=injections))
    return tokens, masks["load"][:, -1]


def _probe_features(tokens: Dict[str, torch.Tensor], hidden_loads: torch.Tensor) -> torch.Tensor:
    """Load token, mean token over observed loads and observed fraction."""
    load_tokens = tokens["load"]
    seen = ~hidden_loads
    pool = load_tokens[seen].mean(0, keepdim=True) if bool(seen.any()) else load_tokens.mean(0, keepdim=True)
    return torch.cat([load_tokens, pool.expand_as(load_tokens),
                      seen.float().mean().expand(load_tokens.shape[0], 1)], dim=-1)


@torch.no_grad()
def _collect_probe_data(model, exp, dataset, device, generator):
    features, targets = [], []
    levels = exp.cfg.hidden_levels
    for index in range(0, len(dataset), PROBE_STRIDE):
        fraction = levels[int(torch.randint(len(levels), (1,), generator=generator).item())]
        observed = random_unobserved(exp.topology.n_sub, fraction, generator)
        sample = dataset[index].to(device)
        tokens, hidden = _tokens(model, exp, sample, observed, device)
        if bool(hidden.any()):
            features.append(_probe_features(tokens, hidden)[hidden].cpu())
            targets.append(sample["load"].x[hidden, model.lookback_max - 1].cpu())
    return torch.cat(features), torch.cat(targets).unsqueeze(-1)


def fit_imputation_probe(model: STGGS, exp: Experiment, dataset, device) -> nn.Module:
    """Trains a new load-imputation head on the frozen backbone."""
    generator = torch.Generator().manual_seed(exp.cfg.seed)
    features, targets = _collect_probe_data(model, exp, dataset, device, generator)
    features, targets = features.to(device), targets.to(device)
    hidden = 128
    net = nn.Sequential(nn.Linear(features.shape[-1], hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                        nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n_samples = len(features)
    for _ in range(PROBE_EPOCHS):
        order = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, PROBE_BATCH):
            batch = order[start:start + PROBE_BATCH]
            optimizer.zero_grad()
            nn.functional.mse_loss(net(features[batch]), targets[batch]).backward()
            optimizer.step()
    net.eval()
    return net


def _state_errors(vm, va, true_vm, true_va, pq_buses, source, target) -> Dict[str, float]:
    """Vm MAE over PQ buses and angle-difference MAE over all branches, in degrees."""
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    difference = wrap(np.deg2rad(va[source] - va[target])) - wrap(np.deg2rad(true_va[source] - true_va[target]))
    return {"vm_mae": float(np.abs(vm[pq_buses] - true_vm[pq_buses]).mean()),
            "dtheta_mae_deg": float(np.rad2deg(np.abs(wrap(difference))).mean())}


class Harness:
    """Maps between the model's bus order and the solver's and holds the true operating points."""

    def __init__(self, exp: Experiment, labels: PowerFlowLabels):
        cfg = exp.cfg
        network = build_network(cfg.label_case_file)
        self.exp = exp
        self.labels = labels
        self.solver = Solver(network)
        self.n_bus = self.solver.n_bus
        self.bus_position = network.bus_position
        self.load_p, self.load_q, _ = read_bus_series(cfg.load_mw_csv, cfg.load_mvar_csv, cfg.renewable_csv)

        topology = exp.topology
        self.sub_to_case = np.array([network.bus_position[int(b)] for b in topology.sub_name])
        self.pq_buses = np.zeros(self.n_bus, dtype=bool)
        self.pq_buses[self.sub_to_case[np.asarray(topology.sub_bus_type) == 1]] = True
        self.branch_source = self.sub_to_case[np.asarray(topology.line_or_to_subid)]
        self.branch_target = self.sub_to_case[np.asarray(topology.line_ex_to_subid)]
        self.load_case_position = self.sub_to_case[np.asarray(topology.load_to_subid)]

        columns = exp.target_columns["load"]
        self.load_mean = np.array([exp.scaler.stats[c][0] for c in columns], dtype=np.float64)
        self.load_std = np.array([exp.scaler.stats[c][1] for c in columns], dtype=np.float64)

        train_rows = [r for r in (labels.lookup(w.hist_ts) for w in exp.windows["train"]) if r >= 0]
        mean_p = self.load_p.iloc[train_rows].to_numpy().mean(axis=0)
        mean_q = self.load_q.reindex(columns=self.load_p.columns).iloc[train_rows].to_numpy().mean(axis=0)
        self.train_mean_p = np.zeros(self.n_bus)
        self.power_factor = np.zeros(self.n_bus)
        for bus, p, q in zip(self.load_p.columns, mean_p, mean_q):
            position = network.bus_position[int(bus)]
            self.train_mean_p[position] = p
            if abs(p) > 1e-9:
                self.power_factor[position] = q / p

    def _to_bus_vector(self, series) -> np.ndarray:
        vector = np.zeros(self.n_bus)
        for bus, value in series.items():
            vector[self.bus_position[int(bus)]] = value
        return vector

    def true_demand(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._to_bus_vector(self.load_p.iloc[row]), self._to_bus_vector(self.load_q.iloc[row])

    def true_generation(self, row: int, demand: np.ndarray) -> np.ndarray:
        generation = np.zeros(self.n_bus)
        generation[self.sub_to_case] = self.labels.p_inj[row]
        return generation + demand

    def true_state(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        vm, va = np.zeros(self.n_bus), np.zeros(self.n_bus)
        vm[self.sub_to_case] = self.labels.vm[row]
        va[self.sub_to_case] = self.labels.va[row]
        return vm, va

    def loads_to_buses(self, per_load_mw: np.ndarray) -> np.ndarray:
        vector = np.zeros(self.n_bus)
        np.add.at(vector, self.load_case_position, per_load_mw)
        return vector

    def check_ceiling(self, row: int) -> None:
        """Stops when solving the true loads does not reproduce the stored labels."""
        demand_p, demand_q = self.true_demand(row)
        converged, vm, _ = self.solver.solve(demand_p, demand_q, self.true_generation(row, demand_p))
        if not converged:
            raise SystemExit("the solve with the true loads did not converge")
        error = float(np.abs(vm[self.sub_to_case] - self.labels.vm[row]).max())
        print(f"solving the true loads reproduces the labels to {error:.2e} p.u.")
        if error > CEILING_TOLERANCE_PU:
            raise SystemExit(f"the true-load solve differs from the labels by {error:.2e} p.u.")


@torch.no_grad()
def _estimate_loads(method: str, model, probe, harness: Harness, sample, observed, device) -> np.ndarray:
    """Per-bus active demand estimate in MW."""
    if method == "per_bus_mean":
        return harness.train_mean_p
    exp = harness.exp
    tokens, hidden = _tokens(model, exp, sample, observed, device)
    if method == "model":
        normalised = model.reconstruct(tokens, model.lookback_max)["load"][:, -1, 0]
    else:
        normalised = probe(_probe_features(tokens, hidden)).squeeze(-1)
    return harness.loads_to_buses(normalised.cpu().numpy() * harness.load_std + harness.load_mean)


def evaluate_level(fraction: float, model, probe, harness: Harness, dataset,
                   hours: List[Tuple[int, int]], device) -> Dict[str, Dict[str, float]]:
    """Mean errors over the converged hours of each method at one hidden fraction."""
    generator = torch.Generator().manual_seed(harness.exp.cfg.seed)
    records: Dict[str, list] = {method: [] for method in METHODS}
    for index, row in hours:
        observed = random_unobserved(harness.exp.topology.n_sub, fraction, generator)
        true_p, true_q = harness.true_demand(row)
        generation = harness.true_generation(row, true_p)
        true_vm, true_va = harness.true_state(row)
        hidden = np.zeros(harness.n_bus, dtype=bool)
        hidden[harness.sub_to_case] = ~observed.numpy()
        sample = dataset[index].to(device)

        for method in METHODS:
            estimate = _estimate_loads(method, model, probe, harness, sample, observed, device)
            demand_p, demand_q = true_p.copy(), true_q.copy()
            demand_p[hidden] = estimate[hidden]
            demand_q[hidden] = demand_p[hidden] * harness.power_factor[hidden]
            converged, vm, va = harness.solver.solve(demand_p, demand_q, generation)
            record = {"converged": converged, "load_mae_mw": float(np.abs(demand_p[hidden] - true_p[hidden]).mean())}
            if converged:
                record.update(_state_errors(vm, va, true_vm, true_va, harness.pq_buses,
                                            harness.branch_source, harness.branch_target))
            records[method].append(record)

    summary = {}
    for method, rows in records.items():
        solved = [r for r in rows if r["converged"]]
        summary[method] = {"convergence_rate": len(solved) / max(len(rows), 1)}
        for key in ("load_mae_mw", "vm_mae", "dtheta_mae_deg"):
            if solved:
                summary[method][key] = float(np.mean([r[key] for r in solved]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_arguments(parser)
    args = parser.parse_args()

    cfg = config_from_args(args)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    exp = prepare_experiment(cfg, scaler_path=scaler_file(cfg.run_dir))
    labels = PowerFlowLabels(cfg.labels_dir, exp.topology.sub_name)
    model = load_model(exp, cfg.run_dir)
    harness = Harness(exp, labels)

    dataset = exp.dataset(exp.windows["test"], labels)
    hours = [(i, labels.lookup(w.hist_ts)) for i, w in enumerate(exp.windows["test"])]
    hours = [(i, row) for i, row in hours if row >= 0]
    harness.check_ceiling(hours[0][1])

    probe = fit_imputation_probe(model, exp, exp.dataset(exp.windows["train"], labels), device)
    results = {}
    for fraction in HIDDEN_LEVELS:
        level = str(int(round(fraction * 100)))
        results[level] = evaluate_level(fraction, model, probe, harness, dataset, hours, device)
        for method, values in results[level].items():
            print(f"hidden {level:>3}% {method:<13} " + "  ".join(f"{k} {v:.4f}" for k, v in values.items()))
    write_json(os.path.join(cfg.results_dir, "impute_solve.json"), results)


if __name__ == "__main__":
    main()

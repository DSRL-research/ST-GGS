"""Generates the hourly AC power-flow labels of an ACTIVSg grid.

Conventional generation has no published series, so each dispatchable unit's
case-file operating point is scaled to meet the residual demand, limited by
[Pmin, Pmax] and iteratively rebalanced. Renewables are must-take. The AC power
flow is solved with Newton-Raphson and a distributed slack weighted by generator
headroom. Hours that do not converge are stored as NaN.

Outputs, in the dataset's `pf` folder:

    pf_bus.npz       p_inj, q_inj, q_shunt, vm_pu, va_degree   (hours, buses)
    pf_branch.npz    p_from, q_from, p_to, q_to                (hours, branches)
    pf_dispatch.npz  per-unit conventional dispatch
    pf_meta.json     timestamps, bus ids, convergence flags, Kirchhoff residual

    python -m stggs.power_flow.generate_labels --dataset activsg2000
"""
import argparse
import json
import os
from typing import Tuple

import numpy as np

from stggs.config import DATASETS, make_config
from stggs.power_flow.network import branch_results, build_network, read_bus_series, require_files


class Fleet:
    """Case generators split into must-take renewables and dispatchable units."""

    def __init__(self, case, renewable_buses, bus_position):
        gens = case.gen
        self.bus = gens["GEN_BUS"].astype(int).to_numpy()
        self.bus_index = np.array([bus_position[b] for b in self.bus])
        self.pmax = gens["PMAX"].astype(float).to_numpy()
        self.pmin = gens["PMIN"].astype(float).to_numpy()
        self.pg = gens["PG"].astype(float).to_numpy()
        self.in_service = gens["GEN_STATUS"].astype(float).to_numpy() > 0
        self.is_renewable = np.isin(self.bus, list(renewable_buses))
        self.dispatchable = np.where(~self.is_renewable)[0]

    def dispatch(self, residual_mw: float) -> Tuple[np.ndarray, np.ndarray]:
        """(output per unit, committed mask) meeting the residual demand."""
        usable = self.dispatchable[self.in_service[self.dispatchable] & (self.pmax[self.dispatchable] > 0)]
        output = np.zeros(len(self.pmax))
        committed = np.zeros(len(self.pmax), dtype=bool)
        base = self.pg[usable]
        if len(usable) == 0 or base.sum() <= 0:
            return output, committed

        output[usable] = np.clip(base * (residual_mw / base.sum()), self.pmin[usable], self.pmax[usable])
        for _ in range(50):
            error = residual_mw - output[usable].sum()
            if abs(error) < 1e-3:
                break
            room = (self.pmax[usable] - output[usable]) if error > 0 else (output[usable] - self.pmin[usable])
            if room.sum() <= 1e-9:
                break
            output[usable] += np.sign(error) * room * (min(abs(error), room.sum()) / room.sum())
        committed[usable] = output[usable] > 1e-9
        return output, committed


def _kirchhoff_residual(bus_out, branch_out, from_bus, to_bus, converged, n_bus) -> dict:
    """Largest absolute P and Q imbalance over a sample of converged hours."""
    rows = np.where(converged)[0]
    worst_p, worst_q = 0.0, 0.0
    for row in rows[::max(1, len(rows) // 50)]:
        p_bus, q_bus = np.zeros(n_bus), np.zeros(n_bus)
        np.add.at(p_bus, from_bus, branch_out["p_from"][row])
        np.add.at(p_bus, to_bus, branch_out["p_to"][row])
        np.add.at(q_bus, from_bus, branch_out["q_from"][row])
        np.add.at(q_bus, to_bus, branch_out["q_to"][row])
        worst_p = max(worst_p, float(np.abs(bus_out["p_inj"][row] - p_bus).max()))
        worst_q = max(worst_q, float(np.abs(bus_out["q_inj"][row] + bus_out["q_shunt"][row] - q_bus).max()))
    return {"n_hours_checked": int(len(rows)), "max_p_mw": worst_p, "max_q_mvar": worst_q}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--out", default=None, help="output folder (default: the dataset's pf folder)")
    parser.add_argument("--hours", type=int, default=None, help="solve only the first N hours")
    args = parser.parse_args()

    import pandapower as pp

    cfg = make_config(args.dataset, data_root=args.data_root, device="cpu")
    require_files(cfg.label_case_file, cfg.load_mw_csv, cfg.load_mvar_csv, cfg.renewable_csv)
    out_dir = args.out or cfg.labels_dir

    network = build_network(cfg.label_case_file)
    net, case, bus_ids, bus_position = network.net, network.case, network.bus_ids, network.bus_position
    load_p, load_q, renewable = read_bus_series(cfg.load_mw_csv, cfg.load_mvar_csv, cfg.renewable_csv)
    n_hours = len(load_p) if args.hours is None else min(args.hours, len(load_p))
    fleet = Fleet(case, {int(c) for c in renewable.columns}, bus_position)

    n_bus, n_branch, n_unit = len(bus_ids), len(case.branch), len(fleet.pmax)
    bus_out = {k: np.zeros((n_hours, n_bus), dtype=np.float32)
               for k in ("p_inj", "q_inj", "q_shunt", "vm_pu", "va_degree")}
    branch_out = {k: np.zeros((n_hours, n_branch), dtype=np.float32)
                  for k in ("p_from", "q_from", "p_to", "q_to")}
    dispatch_p = np.zeros((n_hours, n_unit), dtype=np.float32)
    dispatch_on = np.zeros((n_hours, n_unit), dtype=bool)
    converged = np.zeros(n_hours, dtype=bool)

    position = np.vectorize(bus_position.get)
    load_bus = position(net.load.bus.to_numpy().astype(int))
    gen_bus = position(net.gen.bus.to_numpy().astype(int))
    sgen_bus = position(net.sgen.bus.to_numpy().astype(int)) if len(net.sgen) else np.array([], dtype=int)
    shunt_bus = position(net.shunt.bus.to_numpy().astype(int)) if len(net.shunt) else np.array([], dtype=int)
    slack_bus = bus_position[int(net.ext_grid.bus.iloc[0])]
    sgen_shares_bus = np.isin(sgen_bus, gen_bus) if len(sgen_bus) else np.array([], dtype=bool)

    for hour in range(n_hours):
        demand_p, demand_q, renewable_p = np.zeros(n_bus), np.zeros(n_bus), np.zeros(n_bus)
        for bus, value in load_p.iloc[hour].items():
            demand_p[bus_position[int(bus)]] = value
        for bus, value in load_q.iloc[hour].items():
            demand_q[bus_position[int(bus)]] = value
        for bus, value in renewable.iloc[hour].items():
            renewable_p[bus_position[int(bus)]] = value

        unit_p, committed = fleet.dispatch(demand_p.sum() - renewable_p.sum())
        dispatch_p[hour], dispatch_on[hour] = unit_p, committed
        generation = renewable_p.copy()
        np.add.at(generation, fleet.bus_index, unit_p)

        net.load.loc[:, "p_mw"] = demand_p[load_bus]
        net.load.loc[:, "q_mvar"] = demand_q[load_bus]
        net.gen.loc[:, "p_mw"] = generation[gen_bus]
        if len(net.sgen):
            net.sgen.loc[:, "p_mw"] = np.where(sgen_shares_bus, 0.0, generation[sgen_bus])

        headroom = np.zeros(n_bus)
        np.add.at(headroom, fleet.bus_index, np.where(committed, fleet.pmax - unit_p, 0.0))
        weights = headroom[gen_bus]
        net.gen.loc[:, "slack_weight"] = weights if weights.sum() > 0 else 1.0

        attempts = (dict(init="results" if hour else "dc"), dict(init="dc"), dict(init="flat"),
                    dict(init="dc", max_iteration=100))
        for options in attempts:
            try:
                pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True, numba=False, **options)
            except Exception:
                continue
            if net.converged:
                converged[hour] = True
                break

        if not converged[hour]:
            for key in bus_out:
                bus_out[key][hour] = np.nan
            for key in branch_out:
                branch_out[key][hour] = np.nan
            continue

        p_generation, q_generation = np.zeros(n_bus), np.zeros(n_bus)
        np.add.at(p_generation, gen_bus, net.res_gen.p_mw.to_numpy())
        np.add.at(q_generation, gen_bus, net.res_gen.q_mvar.to_numpy())
        if len(net.sgen):
            np.add.at(p_generation, sgen_bus, net.res_sgen.p_mw.to_numpy())
            np.add.at(q_generation, sgen_bus, net.res_sgen.q_mvar.to_numpy())
        p_generation[slack_bus] += float(net.res_ext_grid.p_mw.iloc[0])
        q_generation[slack_bus] += float(net.res_ext_grid.q_mvar.iloc[0])

        bus_out["p_inj"][hour] = p_generation - demand_p
        bus_out["q_inj"][hour] = q_generation - demand_q
        if len(shunt_bus):
            shunt_q = np.zeros(n_bus)
            np.add.at(shunt_q, shunt_bus, -net.res_shunt.q_mvar.to_numpy())
            bus_out["q_shunt"][hour] = shunt_q
        state = net.res_bus.loc[bus_ids]
        bus_out["vm_pu"][hour] = state.vm_pu.to_numpy()
        bus_out["va_degree"][hour] = state.va_degree.to_numpy()
        branch_out["p_from"][hour], branch_out["q_from"][hour], branch_out["p_to"][hour], branch_out["q_to"][hour] = (
            branch_results(network))

        if (hour + 1) % 100 == 0 or hour + 1 == n_hours:
            print(f"{hour + 1}/{n_hours} hours, converged {converged[:hour + 1].mean():.1%}")

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "pf_bus.npz"), **bus_out)
    np.savez_compressed(os.path.join(out_dir, "pf_branch.npz"), **branch_out)
    np.savez_compressed(os.path.join(out_dir, "pf_dispatch.npz"), p_mw=dispatch_p, committed=dispatch_on,
                        unit_bus=fleet.bus, unit_pmax=fleet.pmax, unit_pmin=fleet.pmin,
                        is_renewable=fleet.is_renewable)

    from_bus = np.array([bus_position[int(b)] for b in case.branch["F_BUS"].astype(int)])
    to_bus = np.array([bus_position[int(b)] for b in case.branch["T_BUS"].astype(int)])
    meta = {
        "case": cfg.label_case_file,
        "hours": int(n_hours),
        "timestamps": [str(t) for t in load_p.index[:n_hours]],
        "bus_ids": bus_ids.tolist(),
        "converged": converged.tolist(),
        "convergence_rate": float(converged.mean()),
        "kcl_residual": _kirchhoff_residual(bus_out, branch_out, from_bus, to_bus, converged, n_bus),
        "dispatch_mode": "scale",
    }
    with open(os.path.join(out_dir, "pf_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out_dir}: converged {converged.mean():.1%} of {n_hours} hours")


if __name__ == "__main__":
    main()

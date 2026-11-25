"""Direct state estimation under partial observability (Table 2, SE columns).

Evaluates the jointly trained state heads and probe heads trained from scratch on
the frozen backbone. Vm is scored at PQ buses and the angle difference over all
branches.

    python -m stggs.evaluation.state_estimation --dataset activsg2000 --variant st_ggs
"""
import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from stggs.config import add_experiment_arguments, config_from_args
from stggs.data.labels import PowerFlowLabels
from stggs.data.observability import hidden_node_mask, injection_channels, random_unobserved
from stggs.data.pipeline import Experiment, prepare_experiment, scaler_file
from stggs.evaluation.common import load_model, write_json
from stggs.models.st_ggs import STGGS

HIDDEN_LEVELS = (0.0, 0.5, 1.0)
PROBE_STRIDE = 6
PROBE_EPOCHS = 60


class StateProbe(nn.Module):
    """Vm from each bus token and the angle difference from each branch's endpoint tokens."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.vm = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.dtheta = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, tokens: torch.Tensor, source: torch.Tensor, target: torch.Tensor):
        vm = self.vm(tokens).squeeze(-1)
        return vm, self.dtheta(torch.cat([tokens[source], tokens[target]], dim=-1)).squeeze(-1)


def _substation_tokens(model: STGGS, exp: Experiment, sample, observed: torch.Tensor, device):
    """Tokens computed with hidden buses and their attached nodes masked."""
    topology = exp.topology
    masks = hidden_node_mask(observed, topology.load_to_subid, topology.gen_to_subid, model.lookback_max,
                             topology.n_load, topology.n_gen, device=device)
    injections = injection_channels(sample["substation"].pf_inj, observed.to(device))
    return model.tokens(*model.encode(sample, mask_dict=masks, injections=injections))


@torch.no_grad()
def _collect_probe_data(model, exp, dataset, device, generator):
    """Tokens and state targets of the training windows, with hidden fractions drawn per window."""
    tokens, vm, dtheta = [], [], []
    levels = exp.cfg.hidden_levels
    for index in range(0, len(dataset), PROBE_STRIDE):
        sample = dataset[index].to(device)
        store = sample["substation"]
        if not bool(store.state_valid.any()):
            continue
        fraction = levels[int(torch.randint(len(levels), (1,), generator=generator).item())]
        observed = random_unobserved(exp.topology.n_sub, fraction, generator)
        tokens.append(_substation_tokens(model, exp, sample, observed, device)["substation"].cpu())
        vm.append(store.y_vm.cpu())
        dtheta.append(store.y_dtheta.cpu())
    return tokens, vm, dtheta


def fit_state_probe(model: STGGS, exp: Experiment, dataset, device) -> StateProbe:
    """Trains a `StateProbe` on training windows under the evaluation masking."""
    generator = torch.Generator().manual_seed(exp.cfg.seed)
    tokens, vm, dtheta = _collect_probe_data(model, exp, dataset, device, generator)

    probe = StateProbe(tokens[0].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    vm_scale = float(torch.cat(vm).std()) or 1.0
    dtheta_scale = float(torch.cat(dtheta).std()) or 1.0
    source, target = model.physics.f, model.physics.t
    for _ in range(PROBE_EPOCHS):
        for index in torch.randperm(len(tokens)):
            optimizer.zero_grad()
            vm_hat, dtheta_hat = probe(tokens[index].to(device), source, target)
            loss = (nn.functional.mse_loss(vm_hat / vm_scale, vm[index].to(device) / vm_scale)
                    + nn.functional.mse_loss(dtheta_hat / dtheta_scale, dtheta[index].to(device) / dtheta_scale))
            loss.backward()
            optimizer.step()
    probe.eval()
    return probe


def _errors(vm_hat: np.ndarray, vm: np.ndarray, dtheta_hat: np.ndarray, dtheta: np.ndarray,
            pq_buses: np.ndarray) -> Dict[str, float]:
    return {"vm_mae": float(np.abs(vm_hat - vm)[:, pq_buses].mean()),
            "dtheta_mae_deg": float(np.abs(np.rad2deg(dtheta_hat) - np.rad2deg(dtheta)).mean())}


@torch.no_grad()
def sweep(model: STGGS, exp: Experiment, dataset, probe: StateProbe, device) -> Dict[str, Dict]:
    """Errors of the joint and probe heads at each hidden fraction."""
    pq_buses = np.asarray(exp.topology.sub_bus_type) == 1
    source, target = model.physics.f, model.physics.t
    results = {}
    for fraction in HIDDEN_LEVELS:
        generator = torch.Generator().manual_seed(exp.cfg.seed)
        collected: Dict[str, list] = {key: [] for key in ("vm", "dtheta", "joint_vm", "joint_dtheta",
                                                          "probe_vm", "probe_dtheta")}
        for index in range(len(dataset)):
            sample = dataset[index].to(device)
            store = sample["substation"]
            if not bool(store.state_valid.any()):
                continue
            observed = random_unobserved(exp.topology.n_sub, fraction, generator)
            tokens = _substation_tokens(model, exp, sample, observed, device)
            joint_vm, joint_dtheta = model.predict_state(tokens)
            probe_vm, probe_dtheta = probe(tokens["substation"], source, target)
            for key, value in (("vm", store.y_vm), ("dtheta", store.y_dtheta), ("joint_vm", joint_vm),
                               ("joint_dtheta", joint_dtheta), ("probe_vm", probe_vm),
                               ("probe_dtheta", probe_dtheta)):
                collected[key].append(value.cpu().numpy())

        arrays = {key: np.stack(value) for key, value in collected.items()}
        results[str(int(round(fraction * 100)))] = {
            head: _errors(arrays[f"{head}_vm"], arrays["vm"], arrays[f"{head}_dtheta"], arrays["dtheta"], pq_buses)
            for head in ("joint", "probe")
        }
    return results


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

    probe = fit_state_probe(model, exp, exp.dataset(exp.windows["train"], labels), device)
    results = sweep(model, exp, exp.dataset(exp.windows["test"], labels), probe, device)
    for level, heads in results.items():
        print(f"hidden {level:>3}%  " + "  ".join(
            f"{head}: Vm {v['vm_mae']:.4f} dtheta {v['dtheta_mae_deg']:.4f}" for head, v in heads.items()))
    write_json(os.path.join(cfg.results_dir, "state_estimation.json"), results)


if __name__ == "__main__":
    main()

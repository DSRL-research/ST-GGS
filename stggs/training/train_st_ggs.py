"""Trains ST-GGS on the joint objective.

Total loss = forecast + masked reconstruction + static contrastive + dynamic
contrastive + state estimation, plus the Kirchhoff term for `st_ggs_kcl`.

    python -m stggs.training.train_st_ggs --dataset activsg2000 --variant st_ggs
"""
import argparse
import json
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stggs.config import FORECAST_NODE_TYPES, Config, add_experiment_arguments, config_from_args
from stggs.data.labels import PowerFlowLabels
from stggs.data.observability import injection_channels, random_unobserved
from stggs.data.pipeline import checkpoint_file, prepare_experiment, scaler_file
from stggs.models.st_ggs import STGGS

LOSS_COMPONENTS = ("total", "forecast", "reconstruction", "static_proto", "static_pair", "dynamic",
                   "state_vm", "state_dtheta", "kcl")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_mask(n_nodes: int, lookback: int, cfg: Config,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """(n_nodes, L) bool mask, True at hidden positions.

    A node is hidden entirely with probability `mask_node_prob`. Otherwise
    contiguous spans of `mask_span_len` steps cover about `mask_span_frac` of it.
    """
    span = max(1, min(cfg.mask_span_len, lookback))
    n_spans = max(1, round(cfg.mask_span_frac * lookback / span))
    positions = torch.arange(lookback)
    mask = torch.zeros(n_nodes, lookback, dtype=torch.bool)
    starts = torch.randint(0, lookback - span + 1, (n_nodes, n_spans), generator=generator)
    for i in range(n_spans):
        start = starts[:, i:i + 1]
        mask |= (positions >= start) & (positions < start + span)
    mask[torch.rand(n_nodes, generator=generator) < cfg.mask_node_prob] = True
    return mask


def _draw_observed(substation, cfg: Config, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Observed-bus mask across a batch, with the hidden fraction drawn per window."""
    n_windows = int(substation.state_valid.shape[0])
    n_bus = substation.pf_inj.shape[0] // n_windows
    levels = cfg.hidden_levels
    masks = [random_unobserved(n_bus, levels[int(torch.randint(len(levels), (1,), generator=generator).item())],
                               generator)
             for _ in range(n_windows)]
    return torch.cat(masks)


def compute_loss(model: STGGS, batch, cfg: Config, scales: Tuple[float, float],
                 generator: Optional[torch.Generator] = None,
                 lookback: Optional[int] = None, horizon: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """All loss terms of one step.

    `generator` makes the masks deterministic, as used for validation. `lookback`
    and `horizon` default to the maxima the windows were built at.
    """
    device = batch["load"].x.device
    lookback = lookback or cfg.lookback
    horizon = horizon or cfg.horizon
    substation = batch["substation"]
    masks = {node_type: generate_mask(batch[node_type].x.shape[0], lookback, cfg, generator).to(device)
             for node_type in FORECAST_NODE_TYPES}
    observed = _draw_observed(substation, cfg, generator).to(device)

    h_own, h_gnn = model.encode(batch, mask_dict=masks, lookback=lookback,
                                injections=injection_channels(substation.pf_inj, observed))
    tokens = model.tokens(h_own, h_gnn)

    forecasts = model.forecast(tokens, batch, horizon)
    forecast = sum(F.mse_loss(forecasts[t], batch[t].y[:, :horizon]) for t in FORECAST_NODE_TYPES)

    reconstructions = model.reconstruct(tokens, lookback)
    terms = []
    for node_type in FORECAST_NODE_TYPES:
        target, _ = model.split_series(node_type, batch[node_type].x, lookback)
        mask = masks[node_type]
        if mask.any():
            terms.append(F.mse_loss(reconstructions[node_type][mask], target[mask]))
    reconstruction = torch.stack(terms).mean() if terms else torch.zeros((), device=device)

    state_vm = state_dtheta = kcl = torch.zeros((), device=device)
    valid = substation.state_valid
    if bool(valid.any()):
        vm_scale, dtheta_scale = scales
        vm, dtheta = model.predict_state(tokens)
        bus_valid = valid.repeat_interleave(substation.y_vm.shape[0] // len(valid))
        branch_valid = valid.repeat_interleave(substation.y_dtheta.shape[0] // len(valid))
        state_vm = F.mse_loss(vm[bus_valid] / vm_scale, substation.y_vm[bus_valid] / vm_scale)
        state_dtheta = F.mse_loss(dtheta[branch_valid] / dtheta_scale,
                                  substation.y_dtheta[branch_valid] / dtheta_scale)
        if cfg.kcl_weight > 0:
            residual_p, residual_q = model.physics.kcl_residual(
                vm[bus_valid], dtheta[branch_valid], substation.p_inj_mw[bus_valid],
                substation.q_inj_mvar[bus_valid], substation.q_shunt[bus_valid])
            at_observed = observed[bus_valid]
            if bool(at_observed.any()):
                kcl = ((residual_p[at_observed] / 100.0) ** 2 + (residual_q[at_observed] / 100.0) ** 2).mean()

    contrastive = model.contrastive_losses(h_gnn, batch.x_dict)
    total = (cfg.forecast_weight * forecast
             + cfg.reconstruction_weight * reconstruction
             + cfg.static_contrastive_weight * (contrastive["static_proto"] + contrastive["static_pair"])
             + cfg.dynamic_contrastive_weight * contrastive["dynamic"]
             + cfg.state_weight * (state_vm + state_dtheta)
             + cfg.kcl_weight * kcl)
    return {"total": total, "forecast": forecast, "reconstruction": reconstruction,
            "state_vm": state_vm, "state_dtheta": state_dtheta, "kcl": kcl, **contrastive}


@torch.no_grad()
def validate(model: STGGS, loader, cfg: Config, scales, device) -> Dict[str, float]:
    """Mean loss terms at the full lookback and horizon, with masks fixed across epochs."""
    model.eval()
    generator = torch.Generator().manual_seed(cfg.seed)
    totals = {key: 0.0 for key in LOSS_COMPONENTS}
    for batch in loader:
        losses = compute_loss(model, batch.to(device), cfg, scales, generator=generator)
        for key in LOSS_COMPONENTS:
            totals[key] += losses[key].item()
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def train(model: STGGS, train_loader, val_loader, cfg: Config, scales) -> None:
    """Runs the training loop, keeping the weights with the lowest validation loss."""
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    context_rng = random.Random(cfg.seed)
    best_loss, best_epoch, history = float("inf"), 0, []

    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = {key: 0.0 for key in LOSS_COMPONENTS}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            losses = compute_loss(model, batch, cfg, scales,
                                  lookback=context_rng.choice(cfg.train_lookbacks),
                                  horizon=context_rng.choice(cfg.train_horizons))
            losses["total"].backward()
            optimizer.step()
            for key in LOSS_COMPONENTS:
                totals[key] += losses[key].item()

        train_terms = {f"train_{k}": v / max(len(train_loader), 1) for k, v in totals.items()}
        val_terms = {f"val_{k}": v for k, v in validate(model, val_loader, cfg, scales, device).items()}
        seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "seconds": seconds, **train_terms, **val_terms})
        pd.DataFrame(history).to_csv(os.path.join(cfg.run_dir, "training_history.csv"), index=False)
        print(f"epoch {epoch:3d} | {seconds:6.1f}s | train {train_terms['train_total']:.4f} | "
              f"val {val_terms['val_total']:.4f}")

        if val_terms["val_total"] < best_loss:
            best_loss, best_epoch = val_terms["val_total"], epoch
            torch.save(model.state_dict(), checkpoint_file(cfg.run_dir))

    total_seconds = sum(row["seconds"] for row in history)
    with open(os.path.join(cfg.run_dir, "sage_best_info.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": cfg.epochs,
                   "total_train_seconds": total_seconds}, f, indent=2)
    print(f"best epoch {best_epoch} (val {best_loss:.4f}), {total_seconds / 60:.1f} min in total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_arguments(parser)
    parser.add_argument("--epochs", type=int, default=None, help="override the number of epochs")
    args = parser.parse_args()

    cfg = config_from_args(args)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    os.makedirs(cfg.run_dir, exist_ok=True)
    set_seed(cfg.seed)

    exp = prepare_experiment(cfg, train_stride=cfg.train_stride)
    exp.scaler.save(scaler_file(cfg.run_dir))

    labels = PowerFlowLabels(cfg.labels_dir, exp.topology.sub_name)
    rows = [labels.lookup(w.hist_ts) for w in exp.windows["train"]]
    scales = labels.target_scales(rows, exp.topology.line_or_to_subid, exp.topology.line_ex_to_subid)

    train_loader = DataLoader(exp.dataset(exp.windows["train"], labels), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(exp.dataset(exp.windows["val"], labels), batch_size=cfg.batch_size, shuffle=False)
    train(STGGS(exp), train_loader, val_loader, cfg, scales)


if __name__ == "__main__":
    main()

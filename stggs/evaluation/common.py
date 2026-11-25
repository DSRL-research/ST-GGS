"""Helpers shared by the evaluation entry points."""
import json
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from stggs.config import FORECAST_NODE_TYPES
from stggs.data.pipeline import Experiment, checkpoint_file
from stggs.data.windows import Window
from stggs.models.st_ggs import STGGS, TOPOLOGY_SIZED_BUFFERS

# Checkpoint entries with no matching module in `STGGS`.
_UNUSED_CHECKPOINT_PREFIXES = ("congestion_head.",)


def load_model(exp: Experiment, run_dir: str, zero_shot: bool = False) -> STGGS:
    """A frozen model with the weights of `run_dir`.

    With `zero_shot`, the checkpoint comes from another grid: buffers whose size
    depends on the grid keep the values built from the target topology.
    """
    device = torch.device(exp.cfg.device)
    state = torch.load(checkpoint_file(run_dir), map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith(_UNUSED_CHECKPOINT_PREFIXES)}
    if zero_shot:
        state = {k: v for k, v in state.items() if k not in TOPOLOGY_SIZED_BUFFERS}

    model = STGGS(exp)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = set(TOPOLOGY_SIZED_BUFFERS) if zero_shot else set()
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def compute_metrics(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """MAE averaged over samples and steps, and R^2 pooled across samples and steps."""
    error = predicted - actual
    total = ((actual - actual.mean()) ** 2).sum()
    return {"mae": float(np.abs(error).mean(axis=0).mean()),
            "r2": float(1 - (error ** 2).sum() / max(total, 1e-8))}


@torch.no_grad()
def forecast_in_mw(model: STGGS, exp: Experiment, windows: Sequence[Window], horizon: int,
                   lookback: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-type (predictions, targets) in MW, ordered window by window, then node by node."""
    device = torch.device(exp.cfg.device)
    dataset = exp.dataset(list(windows))
    predictions = {node_type: [] for node_type in FORECAST_NODE_TYPES}
    targets = {node_type: [] for node_type in FORECAST_NODE_TYPES}
    for index in range(len(dataset)):
        sample = dataset[index].to(device)
        h_own, h_gnn = model.encode(sample, lookback=lookback)
        forecasts = model.forecast(model.tokens(h_own, h_gnn), sample, horizon)
        for node_type in FORECAST_NODE_TYPES:
            predictions[node_type].append(forecasts[node_type].cpu().numpy())
            targets[node_type].append(sample[node_type].y[:, :horizon].cpu().numpy())

    def in_mw(arrays: Dict[str, list]) -> Dict[str, np.ndarray]:
        return {t: exp.scaler.to_physical(np.concatenate(v), exp.target_columns[t]) for t, v in arrays.items()}

    return in_mw(predictions), in_mw(targets)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {path}")

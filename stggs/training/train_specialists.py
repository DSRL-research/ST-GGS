"""Trains the Transformer specialists: one model per horizon and element type.

    python -m stggs.training.train_specialists --dataset activsg2000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stggs.config import (FORECAST_HORIZONS, FORECAST_NODE_TYPES, SPECIALIST_SCHEDULE, Config,
                          add_experiment_arguments, config_from_args)
from stggs.data.pipeline import prepare_experiment
from stggs.data.windows import thin_windows, windows_to_arrays
from stggs.models.transformer import TimeSeriesTransformer


def specialist_file(directory: str, node_type: str, horizon: int) -> str:
    return os.path.join(directory, f"transformer_{node_type}_H{horizon}.pt")


def default_specialists_dir(dataset: str) -> str:
    return os.path.join("runs", dataset, "specialists")


def build_specialist(cfg: Config, horizon: int) -> TimeSeriesTransformer:
    return TimeSeriesTransformer(input_dim=2, horizon=horizon, d_model=cfg.transformer_d_model,
                                 nhead=cfg.transformer_nhead, num_layers=cfg.transformer_layers,
                                 dim_feedforward=cfg.transformer_ff, dropout=cfg.transformer_dropout)


def _loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
                      batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: TimeSeriesTransformer, inputs: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Predictions for `inputs`, in the order given."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(inputs)), batch_size=batch_size, shuffle=False)
    return np.concatenate([model(batch.to(device)).cpu().numpy() for (batch,) in loader])


@torch.no_grad()
def _validation_loss(model, loader, device) -> float:
    model.eval()
    total = sum(F.mse_loss(model(x.to(device)), y.to(device)).item() for x, y in loader)
    return total / max(len(loader), 1)


def train_specialist(model: TimeSeriesTransformer, train_loader, val_loader, cfg: Config,
                     epochs: int, path: str, label: str) -> None:
    """Trains one specialist and writes its best-validation weights and timing to disk."""
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.transformer_lr)
    best_loss, best_epoch, total_seconds = float("inf"), 0, 0.0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        for x, y in tqdm(train_loader, desc=f"{label} epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        val_loss = _validation_loss(model, val_loader, device)
        total_seconds += time.perf_counter() - started
        print(f"{label} epoch {epoch:3d} | val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), path)

    with open(path.replace(".pt", "_info.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": epochs,
                   "total_train_seconds": total_seconds}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_arguments(parser, with_variant=False)
    parser.add_argument("--specialists-dir", default=None,
                        help="output folder (default: runs/<dataset>/specialists)")
    args = parser.parse_args()

    cfg = config_from_args(args)
    directory = args.specialists_dir or default_specialists_dir(cfg.dataset)
    os.makedirs(directory, exist_ok=True)
    torch.manual_seed(cfg.seed)
    exp = prepare_experiment(cfg, train_stride=1)

    for horizon in FORECAST_HORIZONS:
        epochs, stride = SPECIALIST_SCHEDULE[cfg.dataset][horizon]
        train_windows = thin_windows(exp.windows["train"], stride)
        val_windows = thin_windows(exp.windows["val"], stride)
        for node_type in FORECAST_NODE_TYPES:
            x_train, y_train = windows_to_arrays(train_windows, node_type)
            x_val, y_val = windows_to_arrays(val_windows, node_type)
            train_specialist(
                build_specialist(cfg, horizon),
                _loader(x_train, y_train[:, :horizon], cfg.transformer_batch_size, shuffle=True),
                _loader(x_val, y_val[:, :horizon], cfg.transformer_batch_size, shuffle=False),
                cfg, epochs, specialist_file(directory, node_type, horizon), f"{node_type}-H{horizon}")


if __name__ == "__main__":
    main()

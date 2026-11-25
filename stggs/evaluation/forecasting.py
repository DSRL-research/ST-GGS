"""Forecasting performance against the Transformer specialists (Table 1).

Each horizon H is scored on windows spaced H hours apart, so that every test hour
is predicted once at each lead time.

    python -m stggs.evaluation.forecasting --dataset activsg2000 --variant st_ggs
"""
import argparse
import os
from functools import reduce
from math import gcd
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from stggs.config import FORECAST_HORIZONS, FORECAST_NODE_TYPES, add_experiment_arguments, config_from_args
from stggs.data.pipeline import Experiment, prepare_experiment, scaler_file
from stggs.data.windows import Window, thin_windows, windows_to_arrays
from stggs.evaluation.common import compute_metrics, forecast_in_mw, load_model
from stggs.training.train_specialists import (build_specialist, default_specialists_dir, predict,
                                              specialist_file)


def element_groups(exp: Experiment, node_type: str) -> Dict[str, np.ndarray]:
    """{element name: bool mask over the nodes of `node_type`} as reported in Table 1."""
    if node_type == "load":
        return {"load": np.ones(exp.topology.n_load, dtype=bool)}
    return {fuel: exp.topology.gen_type == fuel for fuel in ("solar", "wind")}


def score_elements(exp: Experiment, node_type: str, predicted: np.ndarray,
                   actual: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Metrics per element group for (n_windows * n_nodes, H) arrays."""
    n_nodes = exp.topology.n_load if node_type == "load" else exp.topology.n_gen
    n_windows = predicted.shape[0] // n_nodes
    scores = {}
    for name, nodes in element_groups(exp, node_type).items():
        rows = np.tile(nodes, n_windows)
        scores[name] = {**compute_metrics(predicted[rows], actual[rows]), "n_nodes": int(nodes.sum())}
    return scores


@torch.no_grad()
def score_specialist(exp: Experiment, windows: List[Window], node_type: str, horizon: int, path: str):
    """Element-group metrics of a saved specialist, or None when the checkpoint is absent."""
    if not os.path.exists(path):
        return None
    cfg = exp.cfg
    device = torch.device(cfg.device)
    model = build_specialist(cfg, horizon).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    inputs, targets = windows_to_arrays(windows, node_type)
    columns = exp.target_columns[node_type]
    predicted = exp.scaler.to_physical(predict(model, inputs, cfg.transformer_batch_size, device), columns)
    actual = exp.scaler.to_physical(targets[:, :horizon], columns)
    return score_elements(exp, node_type, predicted, actual)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_arguments(parser)
    parser.add_argument("--specialists-dir", default=None,
                        help="folder of Transformer specialists (default: runs/<dataset>/specialists)")
    args = parser.parse_args()

    cfg = config_from_args(args)
    specialists_dir = args.specialists_dir or default_specialists_dir(cfg.dataset)
    base_stride = reduce(gcd, FORECAST_HORIZONS)
    exp = prepare_experiment(cfg, scaler_path=scaler_file(cfg.run_dir), test_stride=base_stride)
    model = load_model(exp, cfg.run_dir)

    rows = []
    for horizon in FORECAST_HORIZONS:
        windows = thin_windows(exp.windows["test"], horizon // base_stride)
        predicted, actual = forecast_in_mw(model, exp, windows, horizon, lookback=cfg.lookback)
        for node_type in FORECAST_NODE_TYPES:
            st_ggs = score_elements(exp, node_type, predicted[node_type], actual[node_type])
            specialist = score_specialist(exp, windows, node_type, horizon,
                                          specialist_file(specialists_dir, node_type, horizon))
            for element, score in st_ggs.items():
                baseline = specialist[element] if specialist else {}
                rows.append({"horizon": horizon, "element": element, "n_nodes": score["n_nodes"],
                             "st_ggs_mae": score["mae"], "st_ggs_r2": score["r2"],
                             "transformer_mae": baseline.get("mae"), "transformer_r2": baseline.get("r2")})
                print(f"H={horizon:<3} {element:<6} ST-GGS MAE {score['mae']:.3f}  R2 {score['r2']:.3f}"
                      + (f" | T MAE {baseline['mae']:.3f}  R2 {baseline['r2']:.3f}" if baseline else ""))

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, "forecasting.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

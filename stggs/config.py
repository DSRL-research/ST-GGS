"""Hyperparameters of the ST-GGS experiments and shared command-line options."""
import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch

DATASETS = ("activsg2000", "activsg10k")
VARIANTS = ("st_ggs", "st_ggs_kcl")
FORECAST_HORIZONS = (6, 12, 24, 48)
NODE_TYPES = ("substation", "load", "generator")
FORECAST_NODE_TYPES = ("load", "generator")

# Epochs and training-window stride of the Transformer specialists, per horizon.
SPECIALIST_SCHEDULE = {
    "activsg2000": {6: (15, 1), 12: (15, 1), 24: (15, 1), 48: (30, 1)},
    "activsg10k": {6: (12, 1), 12: (12, 1), 24: (12, 1), 48: (30, 3)},
}

_DATASET_SETTINGS = {
    "activsg2000": {
        "case_file": "ACTIVSg2000/case_ACTIVSg2000.m",
        "label_case_file": "ACTIVSg2000/ACTIVSg2000.m",
        "aux_file": "ACTIVSg2000/ACTIVSg2000.aux",
        "load_mw_csv": "ACTIVSg/ACTIVISg2000_load_time_series_MW.csv",
        "load_mvar_csv": "ACTIVSg/ACTIVISg2000_load_time_series_MVAR.csv",
        "renewable_csv": "ACTIVSg/ACTIVISg2000_renewable_time_series_MW.csv",
        "labels_dir": "ACTIVSg2000/pf",
        "batch_size": 16,
        "dynamic_sample_k": 16,
        "static_max_pairs_per_category": 32,
    },
    "activsg10k": {
        "case_file": "ACTIVSg10k/case_ACTIVSg10k.m",
        "label_case_file": "ACTIVSg10k/case_ACTIVSg10k.m",
        "aux_file": "ACTIVSg10k/ACTIVSg10k.aux",
        "load_mw_csv": "ACTIVSg/ACTIVSg10k_load_time_series_MW.csv",
        "load_mvar_csv": "ACTIVSg/ACTIVSg10k_load_time_series_MVAR.csv",
        "renewable_csv": "ACTIVSg/ACTIVISg10k_renewable_time_series_MW.csv",
        "labels_dir": "ACTIVSg10k/pf",
        "batch_size": 4,
        "dynamic_sample_k": 24,
        "static_max_pairs_per_category": 48,
    },
}


@dataclass
class Config:
    dataset: str
    variant: str
    case_file: str
    label_case_file: str
    aux_file: str
    load_mw_csv: str
    load_mvar_csv: str
    renewable_csv: str
    labels_dir: str
    run_dir: str
    results_dir: str
    device: str = "cpu"
    seed: int = 0

    # Windowing and chronological episode split.
    lookback: int = 168
    horizon: int = 48
    train_stride: int = 3
    eval_stride: int = 24
    train_frac: float = 0.7
    val_frac: float = 0.15
    train_lookbacks: Tuple[int, ...] = (24, 72, 168)
    train_horizons: Tuple[int, ...] = FORECAST_HORIZONS

    # Backbone.
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    geo_knn_k: int = 4

    # Temporal encoder.
    temporal_d_model: int = 32
    temporal_nhead: int = 4
    temporal_layers: int = 2
    temporal_ff: int = 64

    # Masked reconstruction.
    mask_node_prob: float = 0.15
    mask_span_frac: float = 0.15
    mask_span_len: int = 12

    # Objective weights.
    forecast_weight: float = 1.0
    reconstruction_weight: float = 1.0
    static_contrastive_weight: float = 0.1
    dynamic_contrastive_weight: float = 0.1
    state_weight: float = 1.0
    kcl_weight: float = 0.0

    # Contrastive objectives.
    projection_dim: int = 32
    static_temperature: float = 0.1
    dynamic_temperature: float = 0.1
    dynamic_sample_k: int = 16
    static_max_pairs_per_category: int = 32
    load_bucket_edges_mw: Tuple[float, ...] = (1.0, 5.0, 20.0, 60.0)

    # State estimation: the fraction of hidden buses is drawn per window.
    hidden_levels: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.8, 1.0)

    # Optimisation.
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100

    # Transformer specialists.
    transformer_d_model: int = 32
    transformer_nhead: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 64
    transformer_dropout: float = 0.1
    transformer_lr: float = 1e-3
    transformer_batch_size: int = 64


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_config(
    dataset: str,
    variant: str = "st_ggs",
    data_root: str = "datasets",
    run_dir: str = None,
    results_dir: str = None,
    device: str = None,
    seed: int = 0,
) -> Config:
    """The configuration of one (dataset, variant) experiment.

    `st_ggs_kcl` differs from `st_ggs` only by the Kirchhoff term in the loss.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}', expected one of {DATASETS}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}', expected one of {VARIANTS}")

    settings = dict(_DATASET_SETTINGS[dataset])
    for key in ("case_file", "label_case_file", "aux_file", "load_mw_csv",
                "load_mvar_csv", "renewable_csv", "labels_dir"):
        settings[key] = os.path.join(data_root, settings[key])

    run_dir = run_dir or os.path.join("runs", dataset, variant)
    return Config(
        dataset=dataset,
        variant=variant,
        run_dir=run_dir,
        results_dir=results_dir or run_dir,
        device=device or default_device(),
        seed=seed,
        kcl_weight=1.0 if variant == "st_ggs_kcl" else 0.0,
        **settings,
    )


def add_experiment_arguments(parser: argparse.ArgumentParser, with_variant: bool = True) -> None:
    """Options shared by every training and evaluation entry point."""
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    if with_variant:
        parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--data-root", default="datasets", help="folder holding the ACTIVSg data")
    parser.add_argument("--run-dir", default=None,
                        help="checkpoint folder (default: runs/<dataset>/<variant>)")
    parser.add_argument("--results-dir", default=None, help="output folder (default: the run folder)")
    parser.add_argument("--device", default=None, help="torch device (default: cuda when available)")
    parser.add_argument("--seed", type=int, default=0)


def config_from_args(args: argparse.Namespace) -> Config:
    return make_config(
        dataset=args.dataset,
        variant=getattr(args, "variant", "st_ggs"),
        data_root=args.data_root,
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        device=args.device,
        seed=args.seed,
    )

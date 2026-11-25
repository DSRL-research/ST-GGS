"""Zero-shot transfer across topologies at H = 48 (Table 3).

A checkpoint trained on the source grid forecasts the target grid's test windows
with no retraining, using a normalisation fitted on the target's training split.
Delta-MAE is relative to the model trained natively on the target grid, scored on
the same windows.

    python -m stggs.evaluation.transfer --source activsg2000 --target activsg10k --variant st_ggs
"""
import argparse
import os

import pandas as pd
import torch

from stggs.config import DATASETS, FORECAST_NODE_TYPES, VARIANTS, make_config
from stggs.data.pipeline import prepare_experiment
from stggs.evaluation.common import compute_metrics, forecast_in_mw, load_model

HORIZON = 48


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=DATASETS)
    parser.add_argument("--target", required=True, choices=DATASETS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--source-run-dir", default=None, help="default: runs/<source>/<variant>")
    parser.add_argument("--target-run-dir", default=None, help="default: runs/<target>/<variant>")
    parser.add_argument("--results-dir", default=None, help="default: the target run folder")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = make_config(args.target, args.variant, data_root=args.data_root, run_dir=args.target_run_dir,
                      results_dir=args.results_dir, device=args.device)
    source_run = args.source_run_dir or os.path.join("runs", args.source, args.variant)
    torch.manual_seed(cfg.seed)
    exp = prepare_experiment(cfg)
    windows = exp.windows["test"]

    transferred = load_model(exp, source_run, zero_shot=True)
    native = load_model(exp, cfg.run_dir)
    transfer_predicted, actual = forecast_in_mw(transferred, exp, windows, HORIZON, lookback=cfg.lookback)
    native_predicted, _ = forecast_in_mw(native, exp, windows, HORIZON, lookback=cfg.lookback)

    rows = []
    for node_type in FORECAST_NODE_TYPES:
        transfer_score = compute_metrics(transfer_predicted[node_type], actual[node_type])
        native_score = compute_metrics(native_predicted[node_type], actual[node_type])
        rows.append({"element": node_type, "mae": transfer_score["mae"], "r2": transfer_score["r2"],
                     "delta_mae": transfer_score["mae"] - native_score["mae"]})
        print(f"{args.source} -> {args.target} {node_type:<10} MAE {transfer_score['mae']:.3f}  "
              f"R2 {transfer_score['r2']:.3f}  delta MAE {rows[-1]['delta_mae']:+.3f}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"transfer_from_{args.source}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

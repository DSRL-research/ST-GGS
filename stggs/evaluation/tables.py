"""Collects the evaluation outputs into the result tables, as CSV and Markdown.

Reads `runs/<dataset>/<variant>/` and `runs/<dataset>/specialists/`, and writes one
file per table to `runs/tables/`. Results that have not been produced yet appear
as empty cells.

    python -m stggs.evaluation.tables
"""
import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from stggs.config import DATASETS, FORECAST_HORIZONS, VARIANTS

_ELEMENT_ROWS = (("load", "Load"), ("solar", "PV Gen."), ("wind", "Wind Gen."))
_TRANSFER_ROWS = (("load", "Load"), ("generator", "Gen."))
_SPECIALIST_INFO = re.compile(r"transformer_(\w+?)_H(\d+)_info\.json$")


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if os.path.exists(path) else None


def _specialist_minutes(runs_root: str, dataset: str) -> Dict[tuple, float]:
    """{(node type, horizon): training minutes} of the Transformer specialists."""
    minutes = {}
    for path in glob.glob(os.path.join(runs_root, dataset, "specialists", "*_info.json")):
        match = _SPECIALIST_INFO.search(os.path.basename(path))
        if match:
            minutes[(match.group(1), int(match.group(2)))] = _read_json(path)["total_train_seconds"] / 60.0
    return minutes


def _write(rows: List[dict], out_dir: str, name: str) -> None:
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    header = "| " + " | ".join(table.columns) + " |"
    separator = "|" + "---|" * len(table.columns)
    lines = [header, separator]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))
                                       for v in row) + " |")
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(out_dir, name)}.{{csv,md}}")


def forecasting_table(runs_root: str, dataset: str) -> List[dict]:
    """Table 1 for one grid, ending with a total-training-time row."""
    per_variant = {v: _read_csv(os.path.join(runs_root, dataset, v, "forecasting.csv")) for v in VARIANTS}
    minutes = _specialist_minutes(runs_root, dataset)

    def lookup(frame, element, horizon):
        if frame is None:
            return None
        match = frame[(frame["element"] == element) & (frame["horizon"] == horizon)]
        return match.iloc[0] if len(match) else None

    rows = []
    for element, label in _ELEMENT_ROWS:
        for horizon in FORECAST_HORIZONS:
            row: Dict = {"element": label, "H": horizon}
            baseline = None
            for variant, frame in per_variant.items():
                score = lookup(frame, element, horizon)
                row[f"{variant}_MAE"] = None if score is None else score["st_ggs_mae"]
                row[f"{variant}_R2"] = None if score is None else score["st_ggs_r2"]
                baseline = score if score is not None else baseline
            row["T_MAE"] = None if baseline is None else baseline["transformer_mae"]
            row["T_R2"] = None if baseline is None else baseline["transformer_r2"]
            specialist_type = "load" if element == "load" else "generator"
            row["T_time_min"] = minutes.get((specialist_type, horizon)) if element != "wind" else None
            rows.append(row)

    total: Dict = {"element": "Total train time (min)", "H": ""}
    for variant in VARIANTS:
        info = _read_json(os.path.join(runs_root, dataset, variant, "sage_best_info.json"))
        total[f"{variant}_MAE"] = None if info is None else info["total_train_seconds"] / 60.0
        total[f"{variant}_R2"] = None
    total.update({"T_MAE": sum(minutes.values()) if minutes else None, "T_R2": None, "T_time_min": None})
    rows.append(total)
    return rows


def state_estimation_table(runs_root: str, dataset: str) -> List[dict]:
    """Table 2 for one grid."""
    direct = {v: _read_json(os.path.join(runs_root, dataset, v, "state_estimation.json")) for v in VARIANTS}
    solved = {v: _read_json(os.path.join(runs_root, dataset, v, "impute_solve.json")) for v in VARIANTS}

    def direct_value(variant, level, head, metric):
        result = direct[variant]
        return None if result is None or level not in result else result[level][head][metric]

    def solved_value(variant, level, method, metric):
        result = solved[variant]
        return None if result is None or level not in result else result[level][method].get(metric)

    rows = []
    for level in ("0", "50", "100"):
        for metric_name, direct_metric, solve_metric in (("Imp. MAE", None, "load_mae_mw"),
                                                         ("Vm", "vm_mae", "vm_mae"),
                                                         ("dtheta", "dtheta_mae_deg", "dtheta_mae_deg")):
            row: Dict = {"hidden_%": level, "metric": metric_name}
            for variant in VARIANTS:
                row[f"SE_{variant}"] = direct_value(variant, level, "joint", direct_metric) if direct_metric else None
                row[f"SE(P)_{variant}"] = direct_value(variant, level, "probe", direct_metric) if direct_metric else None
                row[f"Imp->PF_{variant}"] = solved_value(variant, level, "model", solve_metric)
                row[f"Imp(P)->PF_{variant}"] = solved_value(variant, level, "model_probe", solve_metric)
            row["Mean->PF"] = solved_value(VARIANTS[0], level, "per_bus_mean", solve_metric)
            rows.append(row)
    return rows


def transfer_table(runs_root: str) -> List[dict]:
    """Table 3: zero-shot transfer in both directions."""
    rows = []
    for source, target in zip(DATASETS, reversed(DATASETS)):
        for element, label in _TRANSFER_ROWS:
            row: Dict = {"direction": f"{source}->{target}", "element": label}
            for variant in VARIANTS:
                frame = _read_csv(os.path.join(runs_root, target, variant, f"transfer_from_{source}.csv"))
                score = None if frame is None else frame[frame["element"] == element].iloc[0]
                for column, key in (("MAE", "mae"), ("R2", "r2"), ("dMAE", "delta_mae")):
                    row[f"{variant}_{column}"] = None if score is None else score[key]
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out-dir", default=None, help="default: <runs-root>/tables")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.runs_root, "tables")
    os.makedirs(out_dir, exist_ok=True)
    for dataset in DATASETS:
        _write(forecasting_table(args.runs_root, dataset), out_dir, f"table1_forecasting_{dataset}")
        _write(state_estimation_table(args.runs_root, dataset), out_dir, f"table2_state_estimation_{dataset}")
    _write(transfer_table(args.runs_root), out_dir, "table3_transfer")


if __name__ == "__main__":
    main()

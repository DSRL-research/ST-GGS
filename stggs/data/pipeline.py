"""One shared data path for training and every evaluation."""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from stggs.config import Config
from stggs.data.activsg import Topology, load_activsg
from stggs.data.graph import (CategoryInfo, WindowGraphDataset, add_substation_aggregates,
                              build_graph_inputs, build_node_categories)
from stggs.data.labels import PowerFlowLabels
from stggs.data.scaler import Scaler, scaled_columns
from stggs.data.windows import Window, build_windows, split_episode_ids, usable_episode_ids


@dataclass
class Experiment:
    """A loaded dataset: splits, normalisation and the fixed parts of the graph."""

    cfg: Config
    topology: Topology
    scaler: Scaler
    windows: Dict[str, List[Window]]
    target_columns: Dict[str, List[str]]
    categories: CategoryInfo
    statics: Dict[str, np.ndarray]
    static_widths: Dict[str, int]
    edges: Dict
    edge_features: Dict

    def dataset(self, windows: List[Window], labels: Optional[PowerFlowLabels] = None) -> WindowGraphDataset:
        return WindowGraphDataset(windows, self.topology, self.statics, self.edges,
                                  self.edge_features, labels)


def prepare_experiment(cfg: Config, scaler_path: Optional[str] = None, train_stride: int = 1,
                       test_stride: Optional[int] = None) -> Experiment:
    """Loads the dataset, splits it chronologically and builds the windows.

    `scaler_path` reuses a saved normalisation; otherwise one is fitted on the
    training episodes. `train_stride` applies to training and validation windows,
    `test_stride` to test windows (default: `cfg.eval_stride`).
    """
    topology, frame = load_activsg(cfg)
    frame = add_substation_aggregates(frame, topology)
    train_episodes, val_episodes, test_episodes = split_episode_ids(usable_episode_ids(frame, cfg), cfg)

    train_frame = frame[frame["episode_id"].isin(train_episodes)]
    load_magnitude = train_frame[[f"load_p_{n}" for n in topology.load_name]].mean(axis=0).to_numpy()
    scaler = (Scaler.load(scaler_path) if scaler_path
              else Scaler().fit(train_frame, scaled_columns(frame)))
    frame = scaler.transform(frame)

    built: Dict[int, List[Window]] = {}

    def split(stride: int, episodes) -> List[Window]:
        if stride not in built:
            built[stride] = build_windows(frame, topology, cfg.lookback, cfg.horizon, stride)
        return [w for w in built[stride] if w.episode_id in episodes]

    test_stride = test_stride or cfg.eval_stride
    windows = {
        "train": split(train_stride, train_episodes),
        "val": split(train_stride, val_episodes),
        "test": split(test_stride, test_episodes),
    }
    print(f"{cfg.dataset}: test episodes {sorted(test_episodes)}; windows "
          + ", ".join(f"{name} {len(items)}" for name, items in windows.items()))

    statics, widths, edges, edge_features = build_graph_inputs(topology, cfg)
    return Experiment(
        cfg=cfg,
        topology=topology,
        scaler=scaler,
        windows=windows,
        target_columns={
            "load": [f"load_p_{n}" for n in topology.load_name],
            "generator": [f"prod_p_{n}" for n in topology.gen_name],
        },
        categories=build_node_categories(topology, cfg, load_magnitude),
        statics=statics,
        static_widths=widths,
        edges=edges,
        edge_features=edge_features,
    )


def scaler_file(run_dir: str) -> str:
    return os.path.join(run_dir, "scaler.json")


def checkpoint_file(run_dir: str) -> str:
    return os.path.join(run_dir, "sage_best.pt")

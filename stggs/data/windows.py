"""Lookback/horizon windows and the chronological episode split."""
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from stggs.config import Config
from stggs.data.activsg import Topology


@dataclass
class Window:
    """Arrays are time-major: (lookback, n_nodes) for history, (horizon, n_nodes) for targets."""

    episode_id: str
    load_p_hist: np.ndarray
    load_q_hist: np.ndarray
    prod_p_hist: np.ndarray
    prod_v_hist: np.ndarray
    sub_p_hist: np.ndarray
    sub_g_hist: np.ndarray
    load_p_target: np.ndarray
    prod_p_target: np.ndarray
    hist_ts: np.ndarray
    target_ts: np.ndarray


def build_windows(df: pd.DataFrame, topology: Topology, lookback: int, horizon: int,
                  stride: int) -> List[Window]:
    """Slides a window over each episode. Windows never cross an episode boundary."""
    columns = {
        "load_p": [f"load_p_{n}" for n in topology.load_name],
        "load_q": [f"load_q_{n}" for n in topology.load_name],
        "prod_p": [f"prod_p_{n}" for n in topology.gen_name],
        "prod_v": [f"prod_v_{n}" for n in topology.gen_name],
        "sub_p": [f"subp_{n}" for n in topology.sub_name],
        "sub_g": [f"subg_{n}" for n in topology.sub_name],
    }

    windows: List[Window] = []
    for episode_id, block in df.groupby("episode_id", sort=False):
        block = block.sort_index()
        n_steps = len(block)
        if n_steps < lookback + horizon:
            continue

        arrays = {k: block[v].to_numpy(dtype=np.float32) for k, v in columns.items()}
        timestamps = block.index.to_numpy()
        for start in range(0, n_steps - (lookback + horizon) + 1, stride):
            hist = slice(start, start + lookback)
            target = slice(start + lookback, start + lookback + horizon)
            windows.append(
                Window(
                    episode_id=str(episode_id),
                    load_p_hist=arrays["load_p"][hist],
                    load_q_hist=arrays["load_q"][hist],
                    prod_p_hist=arrays["prod_p"][hist],
                    prod_v_hist=arrays["prod_v"][hist],
                    sub_p_hist=arrays["sub_p"][hist],
                    sub_g_hist=arrays["sub_g"][hist],
                    load_p_target=arrays["load_p"][target],
                    prod_p_target=arrays["prod_p"][target],
                    hist_ts=timestamps[hist],
                    target_ts=timestamps[target],
                )
            )
    return windows


def thin_windows(windows: Sequence[Window], step: int) -> List[Window]:
    """Keeps every `step`-th window of each episode."""
    kept, seen = [], {}
    for window in windows:
        index = seen.get(window.episode_id, 0)
        seen[window.episode_id] = index + 1
        if index % step == 0:
            kept.append(window)
    return kept


def usable_episode_ids(df: pd.DataFrame, cfg: Config) -> List[str]:
    """Episodes long enough to hold at least one full window."""
    counts = df.groupby("episode_id").size()
    return [str(e) for e, n in counts.items() if n >= cfg.lookback + cfg.horizon]


def split_episode_ids(episode_ids: Sequence[str], cfg: Config) -> Tuple[Set[str], Set[str], Set[str]]:
    """Chronological train/validation/test split: test episodes are the latest."""
    episode_ids = sorted(set(episode_ids))
    n_episodes = len(episode_ids)
    n_test = max(1, round(n_episodes * (1 - cfg.train_frac - cfg.val_frac)))
    n_val = max(1, round(n_episodes * cfg.val_frac))
    n_train = n_episodes - n_val - n_test
    if n_train < 1:
        raise ValueError(f"{n_episodes} usable episodes cannot be split into three sets")
    return (set(episode_ids[:n_train]),
            set(episode_ids[n_train:n_train + n_val]),
            set(episode_ids[n_train + n_val:]))


def windows_to_arrays(windows: Sequence[Window], node_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node sequences for a graph-free model.

    Returns X of shape (n_windows * n_nodes, L, 2) and y of shape
    (n_windows * n_nodes, H), ordered window by window, then node by node.
    """
    inputs, targets = [], []
    for window in windows:
        if node_type == "load":
            history = np.stack([window.load_p_hist, window.load_q_hist], axis=-1)
            target = window.load_p_target
        else:
            history = np.stack([window.prod_p_hist, window.prod_v_hist], axis=-1)
            target = window.prod_p_target
        inputs.append(np.swapaxes(history, 0, 1))
        targets.append(target.T)
    return np.concatenate(inputs).astype(np.float32), np.concatenate(targets).astype(np.float32)

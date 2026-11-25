"""Per-node z-score normalisation fitted on the training episodes."""
import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_SCALED_PREFIXES = ("load_p_", "load_q_", "prod_p_", "subp_", "subg_")


@dataclass
class Scaler:
    stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns) -> "Scaler":
        for column in columns:
            std = float(df[column].std())
            self.stats[column] = (float(df[column].mean()), std if std >= 1e-6 else 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column, (mean, std) in self.stats.items():
            if column in df.columns:
                df[column] = (df[column] - mean) / std
        return df

    def to_physical(self, values: np.ndarray, columns) -> np.ndarray:
        """Inverts the scaling of (n_samples, H) values ordered window by window,
        then node by node, with one node per entry of `columns`."""
        mean = np.array([self.stats[c][0] for c in columns], dtype=np.float32)
        std = np.array([self.stats[c][1] for c in columns], dtype=np.float32)
        n_nodes = len(columns)
        grid = values.reshape(values.shape[0] // n_nodes, n_nodes, -1)
        return (grid * std[None, :, None] + mean[None, :, None]).reshape(values.shape)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.stats, f)

    @classmethod
    def load(cls, path: str) -> "Scaler":
        with open(path) as f:
            raw = json.load(f)
        return cls(stats={c: (float(m), float(s)) for c, (m, s) in raw.items()})


def scaled_columns(df: pd.DataFrame) -> list:
    """Columns that are normalised. The capacity-factor channel stays in its own units."""
    return [c for c in df.columns if c.startswith(_SCALED_PREFIXES)]

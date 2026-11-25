"""Cyclical calendar encodings of a sequence of timestamps."""
import numpy as np
import pandas as pd

CALENDAR_DIM = 6


def calendar_features(timestamps) -> np.ndarray:
    """(n_timestamps, 6): sine and cosine of hour of day, day of week and day of year."""
    ts = pd.DatetimeIndex(timestamps)
    hour = ts.hour.to_numpy(dtype=np.float32)
    weekday = ts.dayofweek.to_numpy(dtype=np.float32)
    yearday = ts.dayofyear.to_numpy(dtype=np.float32) - 1.0

    parts = []
    for values, period in ((hour, 24.0), (weekday, 7.0), (yearday, 365.25)):
        theta = 2.0 * np.pi * values / period
        parts.extend([np.sin(theta), np.cos(theta)])
    return np.stack(parts, axis=1).astype(np.float32)

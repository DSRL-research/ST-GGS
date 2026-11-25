"""Sinusoidal positional encoding and the graph-free Transformer baseline."""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Encoder-only Transformer: predicts all H steps from the last-timestep encoding."""

    def __init__(self, input_dim: int, horizon: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.pos_encoding(self.input_proj(x)))
        return self.head(encoded[:, -1, :])

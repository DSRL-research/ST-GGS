"""Horizon-agnostic forecast head."""
import torch
import torch.nn as nn

from stggs.models.transformer import PositionalEncoding


class ForecastHead(nn.Module):
    """Decodes a node token into any number of forecast steps.

    The token is repeated per step, joined with that step's calendar encoding and
    positional encoding, and passed through a small Transformer. No parameter
    depends on the horizon.
    """

    def __init__(self, context_dim: int, aux_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 1, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(context_dim + aux_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, context: torch.Tensor, steps: int, calendar: torch.Tensor) -> torch.Tensor:
        """`context`: (N, D). `calendar`: (N, steps, C). Returns (N, steps)."""
        repeated = context.unsqueeze(1).expand(-1, steps, -1)
        inputs = self.pos_encoding(self.input_proj(torch.cat([repeated, calendar], dim=-1)))
        return self.out_proj(self.encoder(inputs)).squeeze(-1)

"""Per-timestep temporal encoder of a node's history."""
import torch
import torch.nn as nn

from stggs.models.transformer import PositionalEncoding


class TemporalEncoder(nn.Module):
    """Transformer over the lookback, summarised by concatenating mean and last-step pooling.

    Masked positions are replaced by a learned mask token before encoding. Positions
    are sinusoidal, so the same weights accept any lookback length.
    """

    def __init__(self, in_channels: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.out_dim = d_model * 2
        self.value_proj = nn.Linear(in_channels, d_model)
        self.mask_token = nn.Parameter(0.02 * torch.randn(d_model))
        self.pos_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """`sequence`: (N, L, C). `mask`: optional (N, L), True at hidden positions."""
        tokens = self.value_proj(sequence)
        if mask is not None and mask.any():
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        encoded = self.encoder(self.pos_encoding(tokens))
        return torch.cat([encoded.mean(dim=1), encoded[:, -1, :]], dim=-1)

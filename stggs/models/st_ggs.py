"""The spatio-temporal grid GraphSAGE (ST-GGS) model.

A grid element's token is built in two stages. A per-type temporal encoder
summarises the element's history, and the summary is fused with its static
attributes to form h_own. K rounds of heterogeneous message passing with edge
attributes then produce the neighbourhood embedding, and the token concatenates
the two. Heads read the token for forecasting, masked reconstruction and state
estimation.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, MessagePassing

from stggs.config import FORECAST_NODE_TYPES, NODE_TYPES, Config
from stggs.data.calendar_features import CALENDAR_DIM
from stggs.data.pipeline import Experiment
from stggs.models.contrastive import (dynamic_infonce_loss, pairwise_target_similarity,
                                      prototype_loss, sample_candidates,
                                      within_category_pairwise_loss)
from stggs.models.heads import ForecastHead
from stggs.models.physics import BranchPhysics
from stggs.models.temporal import TemporalEncoder

# Buffers whose shape depends on the number of nodes of the grid.
TOPOLOGY_SIZED_BUFFERS = ("category_idx_load", "category_idx_generator", "spec_load",
                          "spec_generator", "load_to_subid", "gen_to_subid")

_INJECTION_CHANNELS = 3


class EdgeSAGEConv(MessagePassing):
    """GraphSAGE convolution with edge-conditioned messages.

    h_v' = W_root h_v + mean_{u in N(v)} MLP([h_u, e_uv])
    """

    def __init__(self, channels: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.msg = nn.Sequential(nn.Linear(channels + edge_dim, channels), nn.ReLU(),
                                 nn.Linear(channels, channels))
        self.lin_root = nn.Linear(channels, channels)

    def forward(self, x, edge_index, edge_attr):
        source, target = x if isinstance(x, (tuple, list)) else (x, x)
        aggregated = self.propagate(edge_index, x=(source, target), edge_attr=edge_attr,
                                    size=(source.size(0), target.size(0)))
        return self.lin_root(target) + aggregated

    def message(self, x_j, edge_attr):
        return self.msg(torch.cat([x_j, edge_attr], dim=-1))


class STGGS(nn.Module):
    def __init__(self, exp: Experiment):
        super().__init__()
        cfg: Config = exp.cfg
        topology = exp.topology
        self.cfg = cfg
        self.lookback_max = cfg.lookback
        self.static_widths = dict(exp.static_widths)
        hidden = cfg.hidden_dim

        self.temporal = nn.ModuleDict({
            node_type: TemporalEncoder(in_channels=2 + CALENDAR_DIM, d_model=cfg.temporal_d_model,
                                       nhead=cfg.temporal_nhead, num_layers=cfg.temporal_layers,
                                       dim_feedforward=cfg.temporal_ff, dropout=cfg.dropout)
            for node_type in NODE_TYPES
        })
        temporal_width = self.temporal["load"].out_dim
        extra = {"substation": _INJECTION_CHANNELS, "load": 0, "generator": 0}
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(temporal_width + exp.static_widths[node_type] + extra[node_type], hidden)
            for node_type in NODE_TYPES
        })

        self.register_buffer("load_to_subid", torch.tensor(topology.load_to_subid, dtype=torch.long))
        self.register_buffer("gen_to_subid", torch.tensor(topology.gen_to_subid, dtype=torch.long))

        self.convs = nn.ModuleList(
            HeteroConv({edge_type: EdgeSAGEConv(hidden, exp.edge_features[edge_type].shape[-1])
                        for edge_type in exp.edges}, aggr="mean")
            for _ in range(cfg.num_layers)
        )
        self.dropout = nn.Dropout(cfg.dropout)

        self.token_dim = hidden * 2
        self.head = nn.ModuleDict({
            node_type: ForecastHead(self.token_dim, CALENDAR_DIM, dropout=cfg.dropout)
            for node_type in FORECAST_NODE_TYPES
        })
        self.recon_head = nn.ModuleDict({
            node_type: nn.Sequential(nn.Linear(self.token_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, cfg.lookback * 2))
            for node_type in FORECAST_NODE_TYPES
        })
        self.pf_head = self._state_head(self.token_dim, hidden)
        self.pf_dtheta_head = self._state_head(2 * self.token_dim, hidden)
        with torch.no_grad():
            self.pf_head[-1].weight.mul_(0.01)
            self.pf_head[-1].bias.fill_(1.0)
            self.pf_dtheta_head[-1].weight.mul_(0.01)
            self.pf_dtheta_head[-1].bias.zero_()
        self.physics = BranchPhysics(topology)

        self.static_proj = nn.ModuleDict()
        self.prototypes = nn.ParameterDict()
        for node_type, (category_index, names, specification) in exp.categories.items():
            self.static_proj[node_type] = nn.Linear(hidden, cfg.projection_dim)
            self.prototypes[node_type] = nn.Parameter(0.1 * torch.randn(len(names), cfg.projection_dim))
            self.register_buffer(f"category_idx_{node_type}", torch.tensor(category_index, dtype=torch.long))
            self.register_buffer(f"spec_{node_type}", torch.tensor(specification, dtype=torch.float32))
        self.dynamic_proj = nn.Linear(hidden, cfg.projection_dim)

    def _state_head(self, input_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(self.cfg.dropout),
                             nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def split_series(self, node_type: str, x: torch.Tensor, lookback: Optional[int] = None):
        """Flat features (N, 2 L_max + F) to a sequence (N, L, 2) and statics (N, F).

        A shorter `lookback` keeps the most recent steps.
        """
        lmax = self.lookback_max
        first, second = x[:, :lmax], x[:, lmax:2 * lmax]
        if lookback is not None and lookback < lmax:
            first, second = first[:, -lookback:], second[:, -lookback:]
        return torch.stack([first, second], dim=-1), x[:, 2 * lmax:]

    def _substation_mask(self, mask_dict: Dict[str, torch.Tensor], n_rows: int) -> torch.Tensor:
        """A substation position is hidden when any attached load or generator is hidden."""
        n_graphs = mask_dict["load"].shape[0] // self.load_to_subid.shape[0]
        n_sub = n_rows // n_graphs
        device = mask_dict["load"].device
        hidden = torch.zeros(n_rows, mask_dict["load"].shape[1], device=device)
        offsets = torch.arange(n_graphs, device=device) * n_sub
        for key, mapping in (("load", self.load_to_subid), ("generator", self.gen_to_subid)):
            rows = offsets.repeat_interleave(mapping.shape[0]) + mapping.repeat(n_graphs)
            hidden.index_add_(0, rows, mask_dict[key].float())
        return hidden > 0

    @staticmethod
    def _per_node(calendar: torch.Tensor, n_rows: int) -> torch.Tensor:
        """Repeats a per-graph calendar tensor (B, S, C) for each node of its graph."""
        return calendar.repeat_interleave(n_rows // calendar.shape[0], dim=0)

    def encode(self, batch, mask_dict: Optional[Dict[str, torch.Tensor]] = None,
               injections: Optional[torch.Tensor] = None,
               lookback: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Returns (h_own, h_gnn), the two per-type embeddings forming the token.

        `mask_dict`: per-type (N, L) bool masks for loads and generators, True at
        hidden positions. `injections`: (n_sub, 3) channels of the state-estimation
        task; zeros, meaning no bus reports, when omitted.
        """
        h_own = {}
        for node_type, x in batch.x_dict.items():
            sequence, statics = self.split_series(node_type, x, lookback)
            calendar = self._per_node(batch.cal_hist, sequence.shape[0])
            if lookback is not None and lookback < calendar.shape[1]:
                calendar = calendar[:, -lookback:, :]
            sequence = torch.cat([sequence, calendar], dim=-1)

            mask = None
            if mask_dict is not None:
                mask = (self._substation_mask(mask_dict, x.shape[0]) if node_type == "substation"
                        else mask_dict[node_type])
            summary = torch.cat([self.temporal[node_type](sequence, mask), statics], dim=1)
            if node_type == "substation":
                channels = injections if injections is not None else summary.new_zeros(
                    summary.shape[0], _INJECTION_CHANNELS)
                summary = torch.cat([summary, channels], dim=1)
            h_own[node_type] = F.relu(self.input_proj[node_type](summary))

        h_gnn = h_own
        for conv in self.convs:
            h_gnn = conv(h_gnn, batch.edge_index_dict, edge_attr_dict=batch.edge_attr_dict)
            h_gnn = {node_type: self.dropout(F.relu(h)) for node_type, h in h_gnn.items()}
        return h_own, h_gnn

    @staticmethod
    def tokens(h_own: Dict, h_gnn: Dict) -> Dict[str, torch.Tensor]:
        return {node_type: torch.cat([h_own[node_type], h_gnn[node_type]], dim=1) for node_type in h_gnn}

    def forecast(self, tokens: Dict, batch, horizon: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Per-type forecasts (N, horizon)."""
        horizon = horizon or self.cfg.horizon
        calendar = batch.cal_future[:, :horizon, :]
        return {node_type: self.head[node_type](tokens[node_type], horizon,
                                                self._per_node(calendar, tokens[node_type].shape[0]))
                for node_type in FORECAST_NODE_TYPES}

    def reconstruct(self, tokens: Dict, lookback: int) -> Dict[str, torch.Tensor]:
        """Per-type reconstruction (N, lookback, 2) of the most recent `lookback` steps."""
        out = {}
        for node_type in FORECAST_NODE_TYPES:
            full = self.recon_head[node_type](tokens[node_type]).view(-1, self.lookback_max, 2)
            out[node_type] = full[:, -lookback:, :] if lookback < self.lookback_max else full
        return out

    def predict_state(self, tokens: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """(Vm per bus, angle difference per case branch), flattened over the batch."""
        token = tokens["substation"]
        vm = self.pf_head(token).squeeze(-1)
        n_bus = self.physics.n_bus
        offsets = (torch.arange(token.shape[0] // n_bus, device=token.device) * n_bus).unsqueeze(1)
        source = (self.physics.f.unsqueeze(0) + offsets).reshape(-1)
        target = (self.physics.t.unsqueeze(0) + offsets).reshape(-1)
        dtheta = self.pf_dtheta_head(torch.cat([token[source], token[target]], dim=-1)).squeeze(-1)
        return vm, dtheta

    def contrastive_losses(self, h_gnn: Dict, x_dict: Dict) -> Dict[str, torch.Tensor]:
        """The static prototype, static pairwise and dynamic contrastive terms."""
        cfg = self.cfg
        prototype_terms, pairwise_terms = [], []
        for node_type in FORECAST_NODE_TYPES:
            embedding = h_gnn[node_type]
            projection = self.static_proj[node_type](embedding)
            category_index = getattr(self, f"category_idx_{node_type}")
            specification = getattr(self, f"spec_{node_type}")
            n_nodes = category_index.shape[0]
            n_windows = embedding.shape[0] // n_nodes
            categories = category_index.repeat(n_windows)
            instances = torch.arange(n_nodes, device=embedding.device).repeat(n_windows)
            prototype_terms.append(prototype_loss(projection, categories, self.prototypes[node_type],
                                                  cfg.static_temperature))
            pairwise_terms.append(within_category_pairwise_loss(
                projection, categories, instances, specification.repeat(n_windows, 1),
                cfg.static_max_pairs_per_category))

        lmax = self.lookback_max
        history = torch.cat([x_dict["load"][:, :lmax], x_dict["generator"][:, :lmax]], dim=0)
        projection = torch.cat([self.dynamic_proj(h_gnn["load"]), self.dynamic_proj(h_gnn["generator"])], dim=0)
        n_total = projection.shape[0]
        candidates = sample_candidates(n_total, min(cfg.dynamic_sample_k, n_total - 1), projection.device)
        dynamic = dynamic_infonce_loss(projection, candidates, pairwise_target_similarity(history, candidates),
                                       cfg.dynamic_temperature)
        return {
            "static_proto": torch.stack(prototype_terms).mean(),
            "static_pair": torch.stack(pairwise_terms).mean(),
            "dynamic": dynamic,
        }

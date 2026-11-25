"""Partial observability: which buses report telemetry and what the model sees."""
from typing import Dict, Optional

import numpy as np
import torch


def random_unobserved(n_bus: int, hidden_fraction: float,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Bool mask with True at observed buses, hiding a uniformly random subset."""
    observed = torch.ones(n_bus, dtype=torch.bool)
    if hidden_fraction <= 0:
        return observed
    if hidden_fraction >= 1:
        return torch.zeros(n_bus, dtype=torch.bool)
    hidden = torch.randperm(n_bus, generator=generator)[:int(round(hidden_fraction * n_bus))]
    observed[hidden] = False
    return observed


def hidden_node_mask(observed: torch.Tensor, load_to_subid, gen_to_subid, lookback: int,
                     n_load: int, n_gen: int, device=None) -> Dict[str, torch.Tensor]:
    """Hides the current hour of every load and generator at an unobserved bus.

    An injection is the sum of the attached loads and generators, so those nodes
    must lose their current value too. Returns {"load": (n_load, L), "generator":
    (n_gen, L)} with True at hidden positions.
    """
    observed = observed.to(torch.bool).cpu()
    masks = {}
    for node_type, mapping, n_nodes in (("load", load_to_subid, n_load),
                                        ("generator", gen_to_subid, n_gen)):
        substation = torch.as_tensor(np.asarray(mapping), dtype=torch.long)
        mask = torch.zeros(n_nodes, lookback, dtype=torch.bool)
        mask[~observed[substation], -1] = True
        masks[node_type] = mask if device is None else mask.to(device)
    return masks


def injection_channels(injections: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """(n_bus, 3): P and Q injection in p.u., zeroed where hidden, plus the observed flag."""
    flag = observed.to(injections.dtype).unsqueeze(-1)
    return torch.cat([injections * flag, flag], dim=-1)

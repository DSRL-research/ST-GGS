"""Static (category) and dynamic (temporal similarity) contrastive objectives."""
import torch
import torch.nn.functional as F


def prototype_loss(projection: torch.Tensor, category_index: torch.Tensor,
                   prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-entropy over the cosine similarity between nodes and category prototypes."""
    if prototypes.shape[0] < 2:
        return projection.new_zeros(())
    logits = (F.normalize(projection, dim=-1) @ F.normalize(prototypes, dim=-1).t()) / temperature
    return F.cross_entropy(logits, category_index)


def within_category_pairwise_loss(projection: torch.Tensor, category_index: torch.Tensor,
                                  instance_index: torch.Tensor, specification: torch.Tensor,
                                  max_pairs_per_category: int) -> torch.Tensor:
    """Pulls together sampled same-category pairs, weighted by specification similarity.

    Pairs of different physical nodes are preferred over the same node seen in
    different windows of the batch.
    """
    z = F.normalize(projection, dim=-1)
    device = projection.device
    weights, similarities = [], []

    for category in category_index.unique():
        members = (category_index == category).nonzero(as_tuple=True)[0]
        n_members = members.numel()
        if n_members < 2:
            continue
        n_draws = max_pairs_per_category * 4
        first = members[torch.randint(0, n_members, (n_draws,), device=device)]
        second = members[torch.randint(0, n_members, (n_draws,), device=device)]
        distinct = first != second
        first, second = first[distinct], second[distinct]
        if first.numel() == 0:
            continue
        cross_instance = (instance_index[first] != instance_index[second]).long()
        keep = torch.argsort(cross_instance, descending=True, stable=True)[:max_pairs_per_category]
        first, second = first[keep], second[keep]

        weights.append(torch.exp(-((specification[first] - specification[second]) ** 2).sum(dim=-1)
                                 / specification.shape[1]))
        similarities.append((z[first] * z[second]).sum(dim=-1))

    if not weights:
        return projection.new_zeros(())
    return (torch.cat(weights) * (1.0 - torch.cat(similarities))).mean()


def sample_candidates(n_nodes: int, k: int, device) -> torch.Tensor:
    """(n_nodes, k) uniformly random candidate indices per anchor, excluding the anchor."""
    indices = torch.randint(0, n_nodes - 1, (n_nodes, k), device=device)
    anchors = torch.arange(n_nodes, device=device).unsqueeze(1)
    return indices + (indices >= anchors).long()


def pairwise_target_similarity(history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between each anchor's active-power history and its candidates'."""
    centred = history - history.mean(dim=1, keepdim=True)
    centred = centred / (centred.norm(dim=1, keepdim=True) + 1e-8)
    return (centred.unsqueeze(1) * centred[candidates]).sum(dim=-1)


def dynamic_infonce_loss(projection: torch.Tensor, candidates: torch.Tensor,
                         target_similarity: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft-label InfoNCE between target correlations and embedding cosine similarities."""
    z = F.normalize(projection, dim=-1)
    predicted = (z.unsqueeze(1) * z[candidates]).sum(dim=-1)
    target = F.softmax(target_similarity / temperature, dim=-1)
    return -(target * F.log_softmax(predicted / temperature, dim=-1)).sum(dim=-1).mean()

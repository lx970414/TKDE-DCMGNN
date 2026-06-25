from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import DCMGNN


def bpr_loss(
    node_embeddings: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    num_users: int,
) -> torch.Tensor:
    user_emb = node_embeddings[users]
    pos_emb = node_embeddings[num_users + pos_items]
    neg_emb = node_embeddings[num_users + neg_items]
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def relation_contrastive_loss(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    model: DCMGNN,
    users: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    per_user = relation_contrastive_losses(outputs, model, users, temperature)
    return per_user.mean()


def relation_contrastive_losses(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    model: DCMGNN,
    users: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    users = torch.unique(users)
    relations = outputs["relations"]
    assert isinstance(relations, dict)
    target = relations[model.target_behavior][users]
    losses: list[torch.Tensor] = []
    labels = torch.arange(users.size(0), device=users.device)

    for behavior in model.behaviors:
        if behavior == model.target_behavior:
            continue
        aux = relations[behavior][users]
        logits = aux @ target.T / temperature
        losses.append(F.cross_entropy(logits, labels, reduction="none"))

    if not losses:
        return target.new_zeros(users.size(0))
    return torch.stack(losses, dim=0).mean(dim=0)


def chain_contrastive_loss(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    model: DCMGNN,
    users: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    per_user = chain_contrastive_losses(outputs, model, users, temperature)
    return per_user.mean()


def chain_contrastive_losses(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    model: DCMGNN,
    users: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    users = torch.unique(users)
    relations = outputs["relations"]
    chain = outputs["chain"]
    assert isinstance(relations, dict)
    assert isinstance(chain, torch.Tensor)
    target = relations[model.target_behavior][users]
    chain_users = chain[users]
    labels = torch.arange(users.size(0), device=users.device)
    logits = chain_users @ target.T / temperature
    return F.cross_entropy(logits, labels, reduction="none")


def l2_regularization(model: DCMGNN) -> torch.Tensor:
    reg = model.user_embedding.weight.norm(2).pow(2)
    reg = reg + model.item_embedding.weight.norm(2).pow(2)
    for transform in model.chain_transforms.values():
        reg = reg + transform.weight.norm(2).pow(2)
    for transform in model.cascade_transforms.values():
        reg = reg + transform.weight.norm(2).pow(2)
    for module in model.channel_gate:
        if hasattr(module, "weight"):
            reg = reg + module.weight.norm(2).pow(2)
    for module in model.contrast_weight_net:
        if hasattr(module, "weight"):
            reg = reg + module.weight.norm(2).pow(2)
    return reg

from __future__ import annotations

import math
import random

import torch

from .data import BehaviorDataset


@torch.no_grad()
def evaluate_topk(
    node_embeddings: torch.Tensor,
    dataset: BehaviorDataset,
    ks: tuple[int, ...] = (5, 10, 20, 40),
    score_chunk_size: int = 4096,
    max_users: int | None = None,
    negatives_per_user: int = 0,
    mask_mode: str = "target",
    item_score_bias: torch.Tensor | None = None,
    seed: int = 42,
) -> dict[str, float]:
    if not dataset.target_test_by_user:
        return {f"Recall@{k}": float("nan") for k in ks} | {
            f"NDCG@{k}": float("nan") for k in ks
        }

    user_emb = node_embeddings[: dataset.num_users]
    item_emb = node_embeddings[dataset.num_users :]
    max_k = max(ks)
    recall_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    user_count = 0

    eval_items = list(dataset.target_test_by_user.items())
    if max_users is not None and max_users > 0:
        eval_items = eval_items[:max_users]
    rng = random.Random(seed)

    for user, positives in eval_items:
        if not positives:
            continue
        candidate_items: list[int] | None = None
        if negatives_per_user > 0:
            blocked = _mask_items(dataset, user, mask_mode) | set(positives)
            candidates = list(positives)
            while len(candidates) < len(positives) + negatives_per_user:
                item = rng.randrange(dataset.num_items)
                if item not in blocked:
                    blocked.add(item)
                    candidates.append(item)
            candidate_items = candidates

        if candidate_items is None:
            scores = _score_items(user_emb[user], item_emb, score_chunk_size)
            if item_score_bias is not None:
                scores = scores + item_score_bias
        else:
            item_index = torch.tensor(candidate_items, dtype=torch.long, device=item_emb.device)
            scores = item_emb[item_index] @ user_emb[user]
            if item_score_bias is not None:
                scores = scores + item_score_bias[item_index]
        train_seen = _mask_items(dataset, user, mask_mode) - set(positives)
        if train_seen and candidate_items is None:
            scores[list(train_seen)] = -torch.inf
        top_positions = torch.topk(scores, k=min(max_k, scores.size(0))).indices.tolist()
        top_items = (
            [candidate_items[position] for position in top_positions]
            if candidate_items is not None
            else top_positions
        )
        positive_set = set(positives)
        user_count += 1

        for k in ks:
            ranked = top_items[:k]
            hits = [1 if item in positive_set else 0 for item in ranked]
            recall_sum[k] += sum(hits) / len(positive_set)
            ndcg_sum[k] += _ndcg(hits, min(len(positive_set), k))

    if user_count == 0:
        return {f"Recall@{k}": float("nan") for k in ks} | {
            f"NDCG@{k}": float("nan") for k in ks
        }

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_sum[k] / user_count
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / user_count
    return metrics


@torch.no_grad()
def evaluate_topk_score_mix(
    weighted_embeddings: list[tuple[torch.Tensor, float]],
    dataset: BehaviorDataset,
    ks: tuple[int, ...] = (5, 10, 20, 40),
    batch_users: int = 256,
    max_users: int | None = None,
    mask_mode: str = "target",
    item_score_bias: torch.Tensor | None = None,
    history_boosts: dict[str, float] | None = None,
) -> dict[str, float]:
    if not dataset.target_test_by_user:
        return {f"Recall@{k}": float("nan") for k in ks} | {
            f"NDCG@{k}": float("nan") for k in ks
        }
    if not weighted_embeddings:
        raise ValueError("weighted_embeddings must contain at least one channel.")

    max_k = max(ks)
    recall_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    user_count = 0
    eval_items = list(dataset.target_test_by_user.items())
    if max_users is not None and max_users > 0:
        eval_items = eval_items[:max_users]
    device = weighted_embeddings[0][0].device

    split_channels = [
        (embeddings[: dataset.num_users], embeddings[dataset.num_users :], weight)
        for embeddings, weight in weighted_embeddings
    ]
    history_by_behavior = {
        behavior: _edges_by_user(edges)
        for behavior, edges in dataset.train_edges.items()
        if history_boosts and history_boosts.get(behavior, 0.0) != 0.0
    }
    for start in range(0, len(eval_items), batch_users):
        batch = eval_items[start : start + batch_users]
        users = torch.tensor([user for user, _ in batch], dtype=torch.long, device=device)
        scores: torch.Tensor | None = None
        for user_emb, item_emb, weight in split_channels:
            part = user_emb[users] @ item_emb.T
            scores = weight * part if scores is None else scores + weight * part
        assert scores is not None
        if item_score_bias is not None:
            scores = scores + item_score_bias.view(1, -1)

        for row, (user, positives) in enumerate(batch):
            train_seen = _mask_items(dataset, user, mask_mode) - set(positives)
            if train_seen:
                scores[row, list(train_seen)] = -torch.inf
            if history_boosts:
                target_seen = set(dataset.target_train_by_user.get(user, set()))
                for behavior, boost in history_boosts.items():
                    if boost == 0.0:
                        continue
                    history_items = history_by_behavior.get(behavior, {}).get(user, set()) - target_seen
                    if history_items:
                        scores[row, list(history_items)] += boost
        top_items_by_row = torch.topk(scores, k=min(max_k, scores.size(1)), dim=1).indices.tolist()
        for top_items, (_, positives) in zip(top_items_by_row, batch):
            positive_set = set(positives)
            if not positive_set:
                continue
            user_count += 1
            for k in ks:
                ranked = top_items[:k]
                hits = [1 if item in positive_set else 0 for item in ranked]
                recall_sum[k] += sum(hits) / len(positive_set)
                ndcg_sum[k] += _ndcg(hits, min(len(positive_set), k))

    if user_count == 0:
        return {f"Recall@{k}": float("nan") for k in ks} | {
            f"NDCG@{k}": float("nan") for k in ks
        }
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_sum[k] / user_count
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / user_count
    return metrics


def _score_items(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    scores = []
    for start in range(0, item_embeddings.size(0), chunk_size):
        chunk = item_embeddings[start : start + chunk_size]
        scores.append(chunk @ user_embedding)
    return torch.cat(scores, dim=0)


def _ndcg(hits: list[int], ideal_hits: int) -> float:
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _mask_items(dataset: BehaviorDataset, user: int, mask_mode: str) -> set[int]:
    if mask_mode == "target":
        return set(dataset.target_train_by_user.get(user, set()))
    if mask_mode == "all":
        return set(dataset.all_train_by_user.get(user, set()))
    if mask_mode == "none":
        return set()
    raise ValueError("mask_mode must be one of: target, all, none.")


def _edges_by_user(edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    by_user: dict[int, set[int]] = {}
    for user, item in edges:
        by_user.setdefault(user, set()).add(item)
    return by_user

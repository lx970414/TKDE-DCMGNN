from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmgnn.config import get_dataset_config
from dcmgnn.data import load_behavior_dataset
from dcmgnn.model import DCMGNN
from evaluate_checkpoint import _parse_behavior_layers, _popularity_score_bias, _select_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan per-user non-target history boosts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-blend", default="relation:cart:0.88,cascade_sum:0.12")
    parser.add_argument("--popularity-beta", type=float, default=0.015)
    parser.add_argument("--popularity-weights", default="view:0.0,cart:0.0,buy:1.0")
    parser.add_argument("--max-eval-users", type=int, default=4096)
    parser.add_argument("--batch-users", type=int, default=128)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint["args"]
    config = get_dataset_config(train_args["dataset"], train_args["data_root"])
    dataset = load_behavior_dataset(config).to(device)
    model = DCMGNN.from_dataset(
        dataset,
        embedding_dim=train_args["embedding_dim"],
        num_layers=train_args["layers"],
        behavior_layers=_parse_behavior_layers(
            train_args.get("behavior_layers", ""), len(config.behaviors)
        ),
        fusion_mode=train_args.get("fusion_mode", "static"),
        prior_behavior=train_args.get("prior_behavior", "") or None,
        prior_alpha=train_args.get("prior_alpha", 1.0),
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    with torch.no_grad():
        outputs = model(dataset)
        weighted = [
            (_select_channel(outputs, channel), weight)
            for channel, weight in _parse_score_blend(args.score_blend)
        ]
        item_bias = _popularity_score_bias(dataset, args.popularity_weights, args.popularity_beta)
        rows = []
        for view_boost in (0.16, 0.20, 0.24, 0.30, 0.40):
            for cart_boost in (0.00, 0.05, 0.10, 0.16, 0.24):
                if view_boost == 0 and cart_boost == 0:
                    continue
                metrics = _evaluate(
                    weighted,
                    dataset,
                    item_bias,
                    view_boost,
                    cart_boost,
                    args.max_eval_users,
                    args.batch_users,
                )
                score = metrics["NDCG@20"] + 0.75 * metrics["NDCG@40"]
                rows.append((score, view_boost, cart_boost, metrics))
    rows.sort(reverse=True)
    for score, view_boost, cart_boost, metrics in rows[: args.top]:
        print(
            f"score={score:.6f} view_boost={view_boost:.4f} cart_boost={cart_boost:.4f} "
            f"Recall@10={metrics['Recall@10']:.6f} NDCG@10={metrics['NDCG@10']:.6f} "
            f"Recall@20={metrics['Recall@20']:.6f} NDCG@20={metrics['NDCG@20']:.6f} "
            f"Recall@40={metrics['Recall@40']:.6f} NDCG@40={metrics['NDCG@40']:.6f}"
        )


def _evaluate(
    weighted_embeddings: list[tuple[torch.Tensor, float]],
    dataset,
    item_bias: torch.Tensor,
    view_boost: float,
    cart_boost: float,
    max_eval_users: int,
    batch_users: int,
    ks: tuple[int, ...] = (5, 10, 20, 40),
) -> dict[str, float]:
    split = [(emb[: dataset.num_users], emb[dataset.num_users :], weight) for emb, weight in weighted_embeddings]
    eval_items = list(dataset.target_test_by_user.items())[:max_eval_users]
    max_k = max(ks)
    recall_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    user_count = 0
    device = split[0][0].device
    view_by_user = _edges_by_user(dataset.train_edges.get("view", set()))
    cart_by_user = _edges_by_user(dataset.train_edges.get("cart", set()))

    for start in range(0, len(eval_items), batch_users):
        batch = eval_items[start : start + batch_users]
        users = torch.tensor([user for user, _ in batch], dtype=torch.long, device=device)
        scores = None
        for user_emb, item_emb, weight in split:
            part = user_emb[users] @ item_emb.T
            scores = weight * part if scores is None else scores + weight * part
        assert scores is not None
        scores = scores + item_bias.view(1, -1)
        for row, (user, positives) in enumerate(batch):
            train_seen = set(dataset.target_train_by_user.get(user, set())) - set(positives)
            if train_seen:
                scores[row, list(train_seen)] = -torch.inf
            view_items = view_by_user.get(user, set()) - set(dataset.target_train_by_user.get(user, set()))
            cart_items = cart_by_user.get(user, set()) - set(dataset.target_train_by_user.get(user, set()))
            if view_boost and view_items:
                scores[row, list(view_items)] += view_boost
            if cart_boost and cart_items:
                scores[row, list(cart_items)] += cart_boost
        top_items_by_row = torch.topk(scores, k=min(max_k, scores.size(1)), dim=1).indices.tolist()
        for top_items, (_, positives) in zip(top_items_by_row, batch):
            positive_set = set(positives)
            if not positive_set:
                continue
            user_count += 1
            for k in ks:
                hits = [1 if item in positive_set else 0 for item in top_items[:k]]
                recall_sum[k] += sum(hits) / len(positive_set)
                ndcg_sum[k] += _ndcg(hits, min(len(positive_set), k))
    return {
        **{f"Recall@{k}": recall_sum[k] / user_count for k in ks},
        **{f"NDCG@{k}": ndcg_sum[k] / user_count for k in ks},
    }


def _edges_by_user(edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    by_user: dict[int, set[int]] = {}
    for user, item in edges:
        by_user.setdefault(user, set()).add(item)
    return by_user


def _parse_score_blend(raw: str) -> list[tuple[str, float]]:
    blend = []
    for part in raw.split(","):
        channel, raw_weight = part.rsplit(":", 1)
        blend.append((channel, float(raw_weight)))
    total = sum(weight for _, weight in blend)
    return [(channel, weight / total) for channel, weight in blend]


def _ndcg(hits: list[int], ideal_hits: int) -> float:
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    main()

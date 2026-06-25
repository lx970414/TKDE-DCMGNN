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
from evaluate_checkpoint import _parse_behavior_layers, _select_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan score-level channel calibration settings.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-users", type=int, default=4096)
    parser.add_argument("--batch-users", type=int, default=256)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--sort-key",
        default="combo",
        choices=["combo", "NDCG@20", "Recall@20", "NDCG@40", "Recall@40"],
        help="Metric used to rank scanned settings.",
    )
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

    max_eval_users = args.max_eval_users if args.max_eval_users > 0 else None
    with torch.no_grad():
        outputs = model(dataset)
        channel_parts = _channel_parts(outputs, dataset.num_users)
        biases = _candidate_biases(dataset)
        rows = []
        for score_name, weights in _score_weight_specs():
            selected = [(channel_parts[name][0], channel_parts[name][1], weight) for name, weight in weights]
            for bias_name, bias in biases:
                metrics = _evaluate_score_mix(
                    selected,
                    dataset,
                    item_score_bias=bias,
                    max_users=max_eval_users,
                    batch_users=args.batch_users,
                )
                rows.append((_scan_score(metrics, args.sort_key), score_name, bias_name, metrics))
    rows.sort(reverse=True)
    for score, score_name, bias_name, metrics in rows[: args.top]:
        print(
            f"scan_score={score:.6f} "
            f"Recall@10={metrics['Recall@10']:.6f} NDCG@10={metrics['NDCG@10']:.6f} "
            f"Recall@20={metrics['Recall@20']:.6f} NDCG@20={metrics['NDCG@20']:.6f} "
            f"Recall@40={metrics['Recall@40']:.6f} NDCG@40={metrics['NDCG@40']:.6f} "
            f"score={score_name} bias={bias_name}"
        )


def _channel_parts(outputs, num_users: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    names = ["relation:cart", "relation:buy", "cascade_sum", "final", "chain"]
    parts = {}
    for name in names:
        embeddings = _select_channel(outputs, name)
        parts[name] = (embeddings[:num_users], embeddings[num_users:])
    return parts


def _score_weight_specs() -> list[tuple[str, tuple[tuple[str, float], ...]]]:
    specs: list[tuple[str, tuple[tuple[str, float], ...]]] = [
        ("cart", (("relation:cart", 1.0),)),
    ]
    for alpha in (0.76, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98):
        specs.append(
            (
                f"cart+cascade:{alpha:.2f}",
                (("relation:cart", alpha), ("cascade_sum", 1.0 - alpha)),
            )
        )
    for cart_w, cascade_w, buy_w in (
        (0.88, 0.08, 0.04),
        (0.90, 0.07, 0.03),
        (0.92, 0.05, 0.03),
        (0.94, 0.04, 0.02),
        (0.90, 0.05, 0.05),
    ):
        specs.append(
            (
                f"cart+cascade+buy:{cart_w:.2f}:{cascade_w:.2f}:{buy_w:.2f}",
                (
                    ("relation:cart", cart_w),
                    ("cascade_sum", cascade_w),
                    ("relation:buy", buy_w),
                ),
            )
        )
    return specs


def _scan_score(metrics: dict[str, float], sort_key: str) -> float:
    if sort_key != "combo":
        return metrics[sort_key]
    return (
        metrics["NDCG@20"]
        + 0.75 * metrics["NDCG@40"]
        + 0.25 * metrics["Recall@20"]
        + 0.15 * metrics["Recall@40"]
    )


def _evaluate_score_mix(
    selected: list[tuple[torch.Tensor, torch.Tensor, float]],
    dataset,
    item_score_bias: torch.Tensor | None,
    max_users: int | None,
    batch_users: int,
    ks: tuple[int, ...] = (5, 10, 20, 40),
) -> dict[str, float]:
    eval_items = list(dataset.target_test_by_user.items())
    if max_users is not None:
        eval_items = eval_items[:max_users]
    max_k = max(ks)
    recall_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    user_count = 0
    device = selected[0][0].device

    for start in range(0, len(eval_items), batch_users):
        batch = eval_items[start : start + batch_users]
        users = torch.tensor([user for user, _ in batch], dtype=torch.long, device=device)
        scores = None
        for user_emb, item_emb, weight in selected:
            part = user_emb[users] @ item_emb.T
            scores = weight * part if scores is None else scores + weight * part
        assert scores is not None
        if item_score_bias is not None:
            scores = scores + item_score_bias.view(1, -1)
        for row, (user, positives) in enumerate(batch):
            train_seen = set(dataset.target_train_by_user.get(user, set())) - set(positives)
            if train_seen:
                scores[row, list(train_seen)] = -torch.inf
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

    return {
        **{f"Recall@{k}": recall_sum[k] / user_count for k in ks},
        **{f"NDCG@{k}": ndcg_sum[k] / user_count for k in ks},
    }


def _ndcg(hits: list[int], ideal_hits: int) -> float:
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _candidate_biases(dataset) -> list[tuple[str, torch.Tensor | None]]:
    beta_grid = {
        "default": (0.015, 0.02, 0.025),
        "buy_heavy": (0.015, 0.02, 0.025),
        "target_only": (0.015, 0.02),
    }
    weight_sets = {
        "default": {"view": 0.2, "cart": 1.0, "buy": 1.5},
        "buy_heavy": {"view": 0.0, "cart": 0.5, "buy": 2.0},
        "target_only": {"view": 0.0, "cart": 0.0, "buy": 1.0},
    }
    biases: list[tuple[str, torch.Tensor | None]] = [("none", None)]
    for weight_name, weights in weight_sets.items():
        base = _standardized_popularity(dataset, weights)
        for beta in beta_grid[weight_name]:
            biases.append((f"{weight_name}:beta={beta}", beta * base))
    return biases


def _standardized_popularity(dataset, behavior_weights: dict[str, float]) -> torch.Tensor:
    device = dataset.pattern_features.device
    counts = torch.zeros(dataset.num_items, dtype=torch.float32, device=device)
    for behavior, edges in dataset.train_edges.items():
        weight = behavior_weights.get(behavior, 1.0)
        if weight == 0 or not edges:
            continue
        item_ids = torch.tensor([item for _, item in edges], dtype=torch.long, device=device)
        counts.scatter_add_(
            0,
            item_ids,
            torch.full((item_ids.numel(),), float(weight), dtype=torch.float32, device=device),
        )
    popularity = torch.log1p(counts)
    return (popularity - popularity.mean()) / popularity.std().clamp_min(1e-6)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import torch

from dcmgnn.config import get_dataset_config
from dcmgnn.data import load_behavior_dataset
from dcmgnn.evaluate import evaluate_topk, evaluate_topk_score_mix
from dcmgnn.model import DCMGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved DCMGNN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-users", type=int, default=0)
    parser.add_argument("--negatives-per-user", type=int, default=0)
    parser.add_argument("--mask-mode", default="target", choices=["target", "all", "none"])
    parser.add_argument("--popularity-beta", type=float, default=0.0)
    parser.add_argument(
        "--score-blend",
        default="",
        help=(
            "Optional score-level blend, e.g. relation:cart:0.98,cascade_sum:0.02. "
            "When set, --channel is ignored."
        ),
    )
    parser.add_argument(
        "--history-boosts",
        default="",
        help="Optional per-user history boosts, e.g. view:0.02,cart:0.04.",
    )
    parser.add_argument(
        "--popularity-weights",
        default="view:0.2,cart:1.0,buy:1.5",
        help="Comma-separated behavior weights for optional log-popularity score bias.",
    )
    parser.add_argument(
        "--channel",
        default="final",
        help=(
            "final, explicit, relation_sum, cascade_sum, chain, relation:<behavior>, all, "
            "or blend|<channel_a>|<channel_b>|<alpha>."
        ),
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
        channels = _channel_names(outputs, args.channel)
        item_score_bias = (
            _popularity_score_bias(dataset, args.popularity_weights, args.popularity_beta)
            if args.popularity_beta != 0
            else None
        )
        history_boosts = _parse_behavior_weights(args.history_boosts) if args.history_boosts else None
        all_metrics = {}
        if args.score_blend:
            if args.negatives_per_user > 0:
                raise ValueError("--score-blend does not support sampled-negative evaluation.")
            blend = _parse_score_blend(args.score_blend)
            weighted_embeddings = [
                (_select_channel(outputs, channel), weight)
                for channel, weight in blend
            ]
            blend_name = "score_blend|" + args.score_blend
            all_metrics[blend_name] = evaluate_topk_score_mix(
                weighted_embeddings,
                dataset,
                max_users=max_eval_users,
                mask_mode=args.mask_mode,
                item_score_bias=item_score_bias,
                history_boosts=history_boosts,
            )
        else:
            for channel in channels:
                embeddings = _select_channel(outputs, channel)
                all_metrics[channel] = evaluate_topk(
                    embeddings,
                    dataset,
                    max_users=max_eval_users,
                    negatives_per_user=args.negatives_per_user,
                    mask_mode=args.mask_mode,
                    item_score_bias=item_score_bias,
                )

    print("saved_metrics", checkpoint.get("metrics", {}))
    for channel, metrics in all_metrics.items():
        print(f"eval_metrics[{channel}]", metrics)
    weights = outputs.get("channel_weights")
    if isinstance(weights, torch.Tensor):
        print("mean_channel_weights", weights.mean(dim=0).detach().cpu().tolist())
    if args.popularity_beta != 0:
        print("popularity_beta", args.popularity_beta)
        print("popularity_weights", args.popularity_weights)
    if args.history_boosts:
        print("history_boosts", args.history_boosts)


def _select_channel(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    channel: str,
) -> torch.Tensor:
    if channel.startswith("blend|"):
        _, left_name, right_name, raw_alpha = channel.split("|", 3)
        alpha = float(raw_alpha)
        left = _select_channel(outputs, left_name)
        right = _select_channel(outputs, right_name)
        return alpha * left + (1.0 - alpha) * right
    if channel.startswith("relation:"):
        behavior = channel.split(":", 1)[1]
        relations = outputs["relations"]
        assert isinstance(relations, dict)
        return relations[behavior]
    value = outputs[channel]
    assert isinstance(value, torch.Tensor)
    return value


def _channel_names(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    channel: str,
) -> list[str]:
    if channel != "all":
        return [channel]
    relations = outputs["relations"]
    assert isinstance(relations, dict)
    return [
        "final",
        "explicit",
        "relation_sum",
        "cascade_sum",
        "chain",
        *[f"relation:{behavior}" for behavior in relations],
    ]


def _parse_behavior_layers(raw: str, behavior_count: int) -> tuple[int, ...] | None:
    if not raw:
        return None
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) != behavior_count:
        raise ValueError(
            f"checkpoint behavior_layers expected {behavior_count} values, got {len(values)}."
        )
    return values


def _parse_score_blend(raw: str) -> list[tuple[str, float]]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--score-blend must contain at least one channel:weight pair.")
    blend: list[tuple[str, float]] = []
    for part in parts:
        channel, raw_weight = part.rsplit(":", 1)
        blend.append((channel, float(raw_weight)))
    total = sum(weight for _, weight in blend)
    if total <= 0:
        raise ValueError("--score-blend weights must sum to a positive value.")
    return [(channel, weight / total) for channel, weight in blend]


def _popularity_score_bias(dataset, raw_weights: str, beta: float) -> torch.Tensor:
    behavior_weights = _parse_behavior_weights(raw_weights)
    device = dataset.pattern_features.device
    counts = torch.zeros(dataset.num_items, dtype=torch.float32, device=device)
    for behavior, edges in dataset.train_edges.items():
        if not edges:
            continue
        weight = behavior_weights.get(behavior, 1.0)
        item_ids = torch.tensor([item for _, item in edges], dtype=torch.long, device=device)
        counts.scatter_add_(
            0,
            item_ids,
            torch.full((item_ids.numel(),), float(weight), dtype=torch.float32, device=device),
        )
    popularity = torch.log1p(counts)
    popularity = (popularity - popularity.mean()) / popularity.std().clamp_min(1e-6)
    return beta * popularity


def _parse_behavior_weights(raw_weights: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in raw_weights.split(","):
        part = part.strip()
        if not part:
            continue
        behavior, raw_value = part.split(":", 1)
        weights[behavior.strip()] = float(raw_value.strip())
    return weights


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmgnn.config import get_dataset_config
from dcmgnn.data import load_behavior_dataset
from dcmgnn.evaluate import evaluate_topk
from dcmgnn.model import DCMGNN
from evaluate_checkpoint import _parse_behavior_layers, _select_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Tmall rank calibration settings.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-users", type=int, default=4096)
    parser.add_argument("--mask-mode", default="target", choices=["target", "all", "none"])
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

    max_eval_users = args.max_eval_users if args.max_eval_users > 0 else None
    with torch.no_grad():
        outputs = model(dataset)
        candidates = _candidate_embeddings(outputs)
        biases = _candidate_biases(dataset)
        rows = []
        for name, embeddings in candidates:
            for bias_name, bias in biases:
                metrics = evaluate_topk(
                    embeddings,
                    dataset,
                    max_users=max_eval_users,
                    mask_mode=args.mask_mode,
                    item_score_bias=bias,
                )
                rows.append((metrics.get("NDCG@20", 0.0), metrics.get("Recall@20", 0.0), name, bias_name, metrics))
        rows.sort(reverse=True)

    for ndcg20, recall20, name, bias_name, metrics in rows[: args.top]:
        print(
            f"NDCG@20={ndcg20:.6f} Recall@20={recall20:.6f} "
            f"Recall@10={metrics.get('Recall@10', 0.0):.6f} "
            f"NDCG@10={metrics.get('NDCG@10', 0.0):.6f} "
            f"channel={name} bias={bias_name}"
        )


def _candidate_embeddings(outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    base_names = ["relation:cart", "relation:buy", "cascade_sum", "final"]
    candidates = [(name, _select_channel(outputs, name)) for name in base_names]
    blend_pairs = [
        ("relation:cart", "relation:buy"),
        ("relation:cart", "cascade_sum"),
        ("relation:cart", "final"),
    ]
    for left_name, right_name in blend_pairs:
        left = _select_channel(outputs, left_name)
        right = _select_channel(outputs, right_name)
        for alpha in (0.80, 0.90, 0.95):
            name = f"blend|{left_name}|{right_name}|{alpha:.2f}"
            candidates.append((name, alpha * left + (1.0 - alpha) * right))
    return candidates


def _candidate_biases(dataset) -> list[tuple[str, torch.Tensor | None]]:
    beta_grid = {
        "default": (0.015, 0.02, 0.025),
        "buy_heavy": (0.01, 0.015, 0.02),
        "target_only": (0.01, 0.015, 0.02),
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

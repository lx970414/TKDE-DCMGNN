from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmgnn.config import get_dataset_config
from dcmgnn.data import build_item_sampling_cdf, load_behavior_dataset
from dcmgnn.evaluate import evaluate_topk_score_mix
from dcmgnn.model import DCMGNN
from evaluate_checkpoint import _parse_behavior_layers, _select_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-channel score calibrator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-eval-users", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--negative-popularity-power", type=float, default=0.5)
    parser.add_argument("--init", default="cart:0.90,cascade_sum:0.10,beta:0.02")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
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
        channels = {
            "cart": _select_channel(outputs, "relation:cart"),
            "cascade_sum": _select_channel(outputs, "cascade_sum"),
            "buy": _select_channel(outputs, "relation:buy"),
        }
        popularity = _standardized_popularity(dataset, {"buy": 1.0})

    calibrator = ScoreCalibrator(args.init).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr)
    train_edges = list(dataset.train_edges[dataset.target_behavior])
    negative_cdf = build_item_sampling_cdf(dataset.item_train_counts, args.negative_popularity_power)
    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        users, pos_items, neg_items = _sample_edges(
            train_edges,
            dataset.target_train_by_user,
            dataset.num_items,
            args.batch_size,
            rng,
            negative_cdf,
            device,
        )
        pos_score = calibrator.score(channels, popularity, users, pos_items, dataset.num_users)
        neg_score = calibrator.score(channels, popularity, users, neg_items, dataset.num_users)
        loss = -F.logsigmoid(pos_score - neg_score).mean() + 0.01 * calibrator.anchor_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.eval_every == 0:
            weights, beta = calibrator.current()
            metrics = evaluate_topk_score_mix(
                [(channels["cart"], weights[0]), (channels["cascade_sum"], weights[1]), (channels["buy"], weights[2])],
                dataset,
                max_users=args.max_eval_users,
                item_score_bias=beta * popularity,
            )
            score = metrics["NDCG@20"] + 0.75 * metrics["NDCG@40"]
            if score > best_score:
                best_score = score
                best_state = (weights, beta, metrics)
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"weights={','.join(f'{w:.4f}' for w in weights)} beta={beta:.4f} "
                f"Recall@20={metrics['Recall@20']:.4f} NDCG@20={metrics['NDCG@20']:.4f} "
                f"Recall@40={metrics['Recall@40']:.4f} NDCG@40={metrics['NDCG@40']:.4f}",
                flush=True,
            )
    if best_state is not None:
        weights, beta, metrics = best_state
        print(
            "best "
            f"weights={','.join(f'{w:.6f}' for w in weights)} beta={beta:.6f} "
            f"Recall@10={metrics['Recall@10']:.6f} NDCG@10={metrics['NDCG@10']:.6f} "
            f"Recall@20={metrics['Recall@20']:.6f} NDCG@20={metrics['NDCG@20']:.6f} "
            f"Recall@40={metrics['Recall@40']:.6f} NDCG@40={metrics['NDCG@40']:.6f}",
            flush=True,
        )


class ScoreCalibrator(nn.Module):
    def __init__(self, init: str) -> None:
        super().__init__()
        values = _parse_init(init)
        weights = torch.tensor(
            [values.get("cart", 0.88), values.get("cascade_sum", 0.12), values.get("buy", 0.0)],
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-6)
        self.logits = nn.Parameter(torch.log(weights.clamp_min(1e-6)))
        self.raw_beta = nn.Parameter(torch.tensor(values.get("beta", 0.015), dtype=torch.float32))
        self.register_buffer("anchor", weights)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.logits, dim=0)
        beta = self.raw_beta.clamp(min=-0.05, max=0.05)
        return weights, beta

    def current(self) -> tuple[list[float], float]:
        weights, beta = self.forward()
        return weights.detach().cpu().tolist(), float(beta.detach().cpu())

    def anchor_loss(self) -> torch.Tensor:
        weights, beta = self.forward()
        return ((weights - self.anchor) ** 2).sum() + beta.pow(2)

    def score(
        self,
        channels: dict[str, torch.Tensor],
        popularity: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
        num_users: int,
    ) -> torch.Tensor:
        weights, beta = self.forward()
        total = None
        for idx, name in enumerate(("cart", "cascade_sum", "buy")):
            embeddings = channels[name]
            user_emb = embeddings[users]
            item_emb = embeddings[num_users + items]
            part = (user_emb * item_emb).sum(dim=-1)
            total = weights[idx] * part if total is None else total + weights[idx] * part
        assert total is not None
        return total + beta * popularity[items]


def _sample_edges(
    train_edges,
    target_train_by_user,
    num_items: int,
    batch_size: int,
    rng: random.Random,
    negative_cdf: tuple[float, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_users: list[int] = []
    pos_items: list[int] = []
    neg_items: list[int] = []
    for _ in range(batch_size):
        user, pos = rng.choice(train_edges)
        blocked = target_train_by_user.get(user, set())
        neg = _draw_negative(num_items, blocked, rng, negative_cdf)
        batch_users.append(user)
        pos_items.append(pos)
        neg_items.append(neg)
    return (
        torch.tensor(batch_users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )


def _draw_negative(
    num_items: int,
    blocked: set[int],
    rng: random.Random,
    negative_cdf: tuple[float, ...],
) -> int:
    for _ in range(1000):
        sample = rng.random()
        lo, hi = 0, len(negative_cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if negative_cdf[mid] < sample:
                lo = mid + 1
            else:
                hi = mid
        if lo not in blocked:
            return lo
    neg = rng.randrange(num_items)
    while neg in blocked:
        neg = rng.randrange(num_items)
    return neg


def _standardized_popularity(dataset, behavior_weights: dict[str, float]) -> torch.Tensor:
    device = dataset.pattern_features.device
    counts = torch.zeros(dataset.num_items, dtype=torch.float32, device=device)
    for behavior, edges in dataset.train_edges.items():
        weight = behavior_weights.get(behavior, 0.0)
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


def _parse_init(raw: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        values[key.strip()] = float(value.strip())
    return values


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dcmgnn.config import get_dataset_config
from dcmgnn.data import build_item_sampling_cdf, load_behavior_dataset, sample_bpr_edges
from dcmgnn.evaluate import evaluate_topk
from dcmgnn.losses import (
    bpr_loss,
    chain_contrastive_loss,
    chain_contrastive_losses,
    l2_regularization,
    relation_contrastive_loss,
    relation_contrastive_losses,
)
from dcmgnn.model import DCMGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCMGNN for multi-behavior recommendation.")
    parser.add_argument("--dataset", default="tmall", choices=["tmall", "Retail_Rocket", "yelp"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means one pass over target positives.")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument(
        "--behavior-layers",
        default="",
        help="Comma-separated per-behavior LightGCN layers, e.g. 3,4,2 for view,cart,buy.",
    )
    parser.add_argument("--fusion-mode", default="static", choices=["static", "dynamic"])
    parser.add_argument("--prior-behavior", default="")
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lambda-rcl", type=float, default=0.05)
    parser.add_argument("--lambda-chain-bpr", type=float, default=0.1)
    parser.add_argument("--lambda-cascade-bpr", type=float, default=0.1)
    parser.add_argument("--lambda-target-bpr", type=float, default=0.5)
    parser.add_argument("--lambda-prior-bpr", type=float, default=0.0)
    parser.add_argument(
        "--negative-mask",
        default="target",
        choices=["target", "all"],
        help="Items excluded from BPR negative sampling.",
    )
    parser.add_argument(
        "--negative-sampling",
        default="uniform",
        choices=["uniform", "popular"],
        help="Negative item sampling distribution for BPR.",
    )
    parser.add_argument(
        "--negative-popularity-power",
        type=float,
        default=0.5,
        help="Power applied to item frequency when --negative-sampling popular is enabled.",
    )
    parser.add_argument("--lambda-distill", type=float, default=0.0)
    parser.add_argument("--distill-channel", default="")
    parser.add_argument("--distill-users", type=int, default=512)
    parser.add_argument("--distill-items", type=int, default=512)
    parser.add_argument("--distill-temperature", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-users", type=int, default=512)
    parser.add_argument("--weighted-rcl", action="store_true")
    parser.add_argument("--contrast-weight-scale", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-eval-users", type=int, default=0, help="0 means evaluate all users.")
    parser.add_argument("--eval-mask-mode", default="target", choices=["target", "all", "none"])
    parser.add_argument("--eval-channel", default="final")
    parser.add_argument(
        "--selection-keys",
        default="Recall@10,NDCG@10,Recall@20",
        help="Comma-separated metric priority used for checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="checkpoints/dcmgnn.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    config = get_dataset_config(args.dataset, args.data_root)
    dataset = load_behavior_dataset(config).to(device)
    rng = random.Random(args.seed)
    selection_keys = _parse_selection_keys(args.selection_keys)

    model = DCMGNN.from_dataset(
        dataset,
        embedding_dim=args.embedding_dim,
        num_layers=args.layers,
        behavior_layers=_parse_behavior_layers(args.behavior_layers, len(config.behaviors)),
        fusion_mode=args.fusion_mode,
        prior_behavior=args.prior_behavior or None,
        prior_alpha=args.prior_alpha,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target_train_size = sum(len(items) for items in dataset.target_train_by_user.values())
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, target_train_size // args.batch_size)
    negative_cdf = (
        build_item_sampling_cdf(dataset.item_train_counts, args.negative_popularity_power)
        if args.negative_sampling == "popular"
        else None
    )

    print(
        f"dataset={config.name} users={dataset.num_users} items={dataset.num_items} "
        f"behaviors={','.join(config.behaviors)} target={config.target_behavior} "
        f"bbps={len(dataset.bbp_adjs)} relation_order={'->'.join(config.relation_order)} "
        f"target_train={target_train_size} steps_per_epoch={steps_per_epoch}",
        flush=True,
    )

    best_score = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        sample_size = args.batch_size * steps_per_epoch
        users, pos_items, neg_items = sample_bpr_edges(
            dataset,
            sample_size,
            device,
            rng,
            negative_mask=args.negative_mask,
            negative_cdf=negative_cdf,
        )
        outputs = model(dataset)
        final = outputs["final"]
        chain = outputs["chain"]
        cascade_sum = outputs["cascade_sum"]
        relations = outputs["relations"]
        assert isinstance(final, torch.Tensor)
        assert isinstance(chain, torch.Tensor)
        assert isinstance(cascade_sum, torch.Tensor)
        assert isinstance(relations, dict)
        target_relation = relations[dataset.target_behavior]
        prior_relation = (
            relations[args.prior_behavior]
            if args.prior_behavior and args.prior_behavior in relations
            else None
        )

        cl_users = users[: args.contrastive_users] if args.contrastive_users > 0 else users
        final_bpr = bpr_loss(final, users, pos_items, neg_items, dataset.num_users)
        chain_bpr = bpr_loss(chain, users, pos_items, neg_items, dataset.num_users)
        cascade_bpr = bpr_loss(cascade_sum, users, pos_items, neg_items, dataset.num_users)
        target_bpr = bpr_loss(target_relation, users, pos_items, neg_items, dataset.num_users)
        prior_bpr = (
            bpr_loss(prior_relation, users, pos_items, neg_items, dataset.num_users)
            if prior_relation is not None
            else final.new_tensor(0.0)
        )
        distill = final.new_tensor(0.0)
        if args.lambda_distill > 0 and args.distill_channel:
            teacher = _select_channel(outputs, args.distill_channel)
            distill = _sampled_listwise_distillation(
                student=final,
                teacher=teacher.detach(),
                users=users,
                pos_items=pos_items,
                neg_items=neg_items,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                max_users=args.distill_users,
                max_items=args.distill_items,
                temperature=args.distill_temperature,
            )
        if args.weighted_rcl:
            unique_cl_users = torch.unique(cl_users)
            rcl_values = relation_contrastive_losses(outputs, model, unique_cl_users, args.temperature)
            ccl_values = chain_contrastive_losses(outputs, model, unique_cl_users, args.temperature)
            weights = model.contrast_weights(
                final,
                chain,
                target_relation,
                unique_cl_users,
                scale=args.contrast_weight_scale,
            )
            rcl = (weights[:, 0] * rcl_values).mean()
            ccl = (weights[:, 1] * ccl_values).mean()
        else:
            rcl = relation_contrastive_loss(outputs, model, cl_users, args.temperature)
            ccl = chain_contrastive_loss(outputs, model, cl_users, args.temperature)
        reg = args.weight_decay * l2_regularization(model) / sample_size
        loss = (
            final_bpr
            + args.lambda_chain_bpr * chain_bpr
            + args.lambda_cascade_bpr * cascade_bpr
            + args.lambda_target_bpr * target_bpr
            + args.lambda_prior_bpr * prior_bpr
            + args.lambda_distill * distill
            + args.lambda_rcl * (rcl + ccl)
            + reg
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_outputs = model(dataset)
                eval_embeddings = _select_channel(eval_outputs, args.eval_channel)
                max_eval_users = args.max_eval_users if args.max_eval_users > 0 else None
                metrics = evaluate_topk(
                    eval_embeddings,
                    dataset,
                    max_users=max_eval_users,
                    mask_mode=args.eval_mask_mode,
                )
            score = _checkpoint_score(metrics, selection_keys)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                _save_checkpoint(args.save_path, model, args, config.name, metrics)
            print(
                _format_log(
                    epoch,
                    loss.item(),
                    final_bpr.item(),
                    rcl.item(),
                    ccl.item(),
                    distill.item(),
                    metrics,
                ),
                flush=True,
            )

    if best_metrics:
        print("best " + _format_metrics(best_metrics), flush=True)


def _save_checkpoint(
    save_path: str,
    model: DCMGNN,
    args: argparse.Namespace,
    dataset_name: str,
    metrics: dict[str, float],
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "dataset": dataset_name,
            "metrics": metrics,
        },
        path,
    )


def _parse_behavior_layers(raw: str, behavior_count: int) -> tuple[int, ...] | None:
    if not raw:
        return None
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) != behavior_count:
        raise ValueError(
            f"--behavior-layers expected {behavior_count} values, got {len(values)}."
        )
    return values


def _select_channel(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    channel: str,
) -> torch.Tensor:
    if channel.startswith("blend|"):
        _, left_name, right_name, raw_alpha = channel.split("|", 3)
        alpha = float(raw_alpha)
        return alpha * _select_channel(outputs, left_name) + (1.0 - alpha) * _select_channel(
            outputs, right_name
        )
    if channel.startswith("relation:"):
        behavior = channel.split(":", 1)[1]
        relations = outputs["relations"]
        assert isinstance(relations, dict)
        return relations[behavior]
    value = outputs[channel]
    assert isinstance(value, torch.Tensor)
    return value


def _sampled_listwise_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    num_users: int,
    num_items: int,
    max_users: int,
    max_items: int,
    temperature: float,
) -> torch.Tensor:
    unique_users = torch.unique(users)
    if max_users > 0 and unique_users.size(0) > max_users:
        unique_users = unique_users[:max_users]
    candidate_items = torch.unique(torch.cat((pos_items, neg_items), dim=0))
    if max_items > 0 and candidate_items.size(0) > max_items:
        candidate_items = candidate_items[:max_items]
    if max_items > 0 and candidate_items.size(0) < max_items:
        needed = max_items - candidate_items.size(0)
        random_items = torch.randint(
            low=0,
            high=num_items,
            size=(needed,),
            device=candidate_items.device,
        )
        candidate_items = torch.unique(torch.cat((candidate_items, random_items), dim=0))
    if unique_users.numel() == 0 or candidate_items.numel() == 0:
        return student.new_tensor(0.0)

    student_users = student[unique_users]
    teacher_users = teacher[unique_users]
    student_items = student[num_users + candidate_items]
    teacher_items = teacher[num_users + candidate_items]
    temp = max(temperature, 1e-6)
    student_scores = student_users @ student_items.T / temp
    teacher_scores = teacher_users @ teacher_items.T / temp
    teacher_probs = F.softmax(teacher_scores, dim=-1)
    return F.kl_div(
        F.log_softmax(student_scores, dim=-1),
        teacher_probs,
        reduction="batchmean",
    ) * (temp * temp)


def _format_log(
    epoch: int,
    loss: float,
    final_bpr: float,
    rcl: float,
    ccl: float,
    distill: float,
    metrics: dict[str, float],
) -> str:
    return (
        f"epoch={epoch:04d} loss={loss:.4f} bpr={final_bpr:.4f} "
        f"relation_cl={rcl:.4f} chain_cl={ccl:.4f} distill={distill:.4f} "
        f"{_format_metrics(metrics)}"
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key in ("Recall@10", "NDCG@10", "Recall@20", "NDCG@20"):
        value = metrics.get(key, float("nan"))
        parts.append(f"{key}={value:.4f}" if value == value else f"{key}=nan")
    return " ".join(parts)


def _parse_selection_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    if not keys:
        raise ValueError("--selection-keys must contain at least one metric name.")
    return keys


def _checkpoint_score(metrics: dict[str, float], selection_keys: tuple[str, ...]) -> tuple[float, ...]:
    values = tuple(metrics.get(key, float("nan")) for key in selection_keys)
    if any(value != value for value in values):
        return tuple(-1.0 for _ in selection_keys)
    return values


if __name__ == "__main__":
    main()

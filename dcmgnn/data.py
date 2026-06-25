from __future__ import annotations

import itertools
import pickle
import random
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import DatasetConfig

Edge = tuple[int, int]


@dataclass
class BehaviorDataset:
    config: DatasetConfig
    num_users: int
    num_items: int
    train_edges: dict[str, set[Edge]]
    test_edges: dict[str, set[Edge]]
    relation_adjs: dict[str, torch.Tensor]
    bbp_adjs: list[torch.Tensor]
    bbp_names: list[tuple[str, ...]]
    pattern_features: torch.Tensor
    target_train_by_user: dict[int, set[int]]
    target_test_by_user: dict[int, set[int]]
    all_train_by_user: dict[int, set[int]]
    item_train_counts: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items

    @property
    def target_behavior(self) -> str:
        return self.config.target_behavior

    def to(self, device: torch.device | str) -> "BehaviorDataset":
        self.relation_adjs = {k: v.to(device) for k, v in self.relation_adjs.items()}
        self.bbp_adjs = [adj.to(device) for adj in self.bbp_adjs]
        self.pattern_features = self.pattern_features.to(device)
        return self


def load_behavior_dataset(config: DatasetConfig) -> BehaviorDataset:
    if _has_behavior_subdirs(config):
        train_edges, test_edges = _load_behavior_subdir_dataset(config)
    else:
        train_edges, test_edges = _load_retail_files(config)

    max_user = max((u for edges in train_edges.values() for u, _ in edges), default=-1)
    max_user = max(max_user, max((u for edges in test_edges.values() for u, _ in edges), default=-1))
    max_item = max((i for edges in train_edges.values() for _, i in edges), default=-1)
    max_item = max(max_item, max((i for edges in test_edges.values() for _, i in edges), default=-1))
    if max_user < 0 or max_item < 0:
        raise ValueError(f"No usable interactions were found under {config.root}.")

    num_users = max_user + 1
    num_items = max_item + 1
    relation_adjs = {
        behavior: _build_norm_adj(edges, num_users, num_items)
        for behavior, edges in train_edges.items()
    }

    bbp_names, bbp_edge_sets = _build_bbp_edge_sets(config.behaviors, train_edges)
    bbp_adjs = [_build_norm_adj(edges, num_users, num_items) for edges in bbp_edge_sets]
    pattern_features = _build_pattern_features(bbp_edge_sets, num_users, num_items)

    target_train_by_user = _edges_by_user(train_edges[config.target_behavior])
    target_test_by_user = _edges_by_user(test_edges.get(config.target_behavior, set()))

    return BehaviorDataset(
        config=config,
        num_users=num_users,
        num_items=num_items,
        train_edges=train_edges,
        test_edges=test_edges,
        relation_adjs=relation_adjs,
        bbp_adjs=bbp_adjs,
        bbp_names=bbp_names,
        pattern_features=pattern_features,
        target_train_by_user=target_train_by_user,
        target_test_by_user=target_test_by_user,
        all_train_by_user=_all_edges_by_user(train_edges),
        item_train_counts=_item_train_counts(train_edges, num_items),
    )


def sample_bpr_batch(
    dataset: BehaviorDataset,
    batch_size: int,
    device: torch.device | str,
    rng: random.Random,
    negative_mask: str = "target",
    negative_cdf: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users = list(dataset.target_train_by_user)
    if not users:
        raise ValueError("The target behavior has no training positives.")
    blocked_by_user = _negative_blocked_by_user(dataset, negative_mask)

    batch_users: list[int] = []
    pos_items: list[int] = []
    neg_items: list[int] = []

    for _ in range(batch_size):
        user = rng.choice(users)
        pos = rng.choice(tuple(dataset.target_train_by_user[user]))
        neg = _sample_negative(dataset.num_items, blocked_by_user.get(user, set()), rng, negative_cdf)
        batch_users.append(user)
        pos_items.append(pos)
        neg_items.append(neg)

    return (
        torch.tensor(batch_users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )


def sample_bpr_edges(
    dataset: BehaviorDataset,
    sample_size: int,
    device: torch.device | str,
    rng: random.Random,
    negative_mask: str = "target",
    negative_cdf: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = list(dataset.train_edges[dataset.target_behavior])
    if not edges:
        raise ValueError("The target behavior has no training positives.")
    blocked_by_user = _negative_blocked_by_user(dataset, negative_mask)
    selected: list[Edge] = []
    while len(selected) < sample_size:
        shuffled = edges[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[: sample_size - len(selected)])

    batch_users: list[int] = []
    pos_items: list[int] = []
    neg_items: list[int] = []
    for user, pos in selected:
        neg = _sample_negative(dataset.num_items, blocked_by_user.get(user, set()), rng, negative_cdf)
        batch_users.append(user)
        pos_items.append(pos)
        neg_items.append(neg)

    return (
        torch.tensor(batch_users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )


def _negative_blocked_by_user(dataset: BehaviorDataset, negative_mask: str) -> dict[int, set[int]]:
    if negative_mask == "target":
        return dataset.target_train_by_user
    if negative_mask == "all":
        return dataset.all_train_by_user
    raise ValueError("negative_mask must be one of: target, all.")


def build_item_sampling_cdf(item_counts: tuple[int, ...], power: float) -> tuple[float, ...]:
    weights = [(count + 1.0) ** power for count in item_counts]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot build item sampling distribution with non-positive total weight.")
    cdf: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cdf.append(running)
    cdf[-1] = 1.0
    return tuple(cdf)


def _sample_negative(
    num_items: int,
    blocked: set[int],
    rng: random.Random,
    negative_cdf: tuple[float, ...] | None = None,
) -> int:
    if len(blocked) >= num_items:
        raise ValueError("Cannot sample a negative item because the user's blocked set covers all items.")
    neg = _draw_item(num_items, rng, negative_cdf)
    tries = 0
    while neg in blocked:
        tries += 1
        neg = _draw_item(num_items, rng, negative_cdf if tries < 1000 else None)
    return neg


def _draw_item(
    num_items: int,
    rng: random.Random,
    negative_cdf: tuple[float, ...] | None = None,
) -> int:
    if negative_cdf is None:
        return rng.randrange(num_items)
    return min(bisect_left(negative_cdf, rng.random()), num_items - 1)


def _has_behavior_subdirs(config: DatasetConfig) -> bool:
    return all((config.root / behavior / "train.txt").exists() for behavior in config.behaviors)


def _load_behavior_subdir_dataset(
    config: DatasetConfig,
) -> tuple[dict[str, set[Edge]], dict[str, set[Edge]]]:
    train_edges: dict[str, set[Edge]] = {}
    test_edges: dict[str, set[Edge]] = {}
    for behavior in config.behaviors:
        behavior_dir = config.root / behavior
        train_edges[behavior] = _read_user_item_list(behavior_dir / "train.txt")
        test_edges[behavior] = _read_user_item_list(behavior_dir / "test.txt")
    return train_edges, test_edges


def _load_retail_files(config: DatasetConfig) -> tuple[dict[str, set[Edge]], dict[str, set[Edge]]]:
    sparse_files = {
        "view": config.root / "train_mat_view.pkl",
        "cart": config.root / "train_mat_cart.pkl",
        "buy": config.root / "train_mat_buy.pkl",
    }
    if all(path.exists() for path in sparse_files.values()):
        train_edges = {
            behavior: _read_sparse_matrix_edges(path)
            for behavior, path in sparse_files.items()
        }
        test_edges = {
            "view": set(),
            "cart": set(),
            "buy": _read_sparse_matrix_edges(config.root / "test_mat.pkl"),
        }
        return train_edges, test_edges

    train_edges = {
        "view": _read_text_or_pickle_edges(config.root / "trn_view"),
        "cart": _read_text_or_pickle_edges(config.root / "trn_cart"),
        "buy": _read_text_or_pickle_edges(config.root / "trn_buy"),
    }
    test_file = config.root / "test.txt"
    target_test = _read_user_item_list(test_file) if test_file.exists() else set()
    return train_edges, {"view": set(), "cart": set(), "buy": target_test}


def _read_text_or_pickle_edges(path: Path) -> set[Edge]:
    try:
        return _read_user_item_list(path)
    except UnicodeDecodeError:
        return _read_sparse_matrix_edges(path)


def _read_sparse_matrix_edges(path: Path) -> set[Edge]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("rb") as handle:
        matrix = pickle.load(handle)
    coo = matrix.tocoo()
    return {(int(user), int(item)) for user, item in zip(coo.row, coo.col)}


def _read_user_item_list(path: Path) -> set[Edge]:
    edges: set[Edge] = set()
    if not path.exists() or path.stat().st_size == 0:
        return edges
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if len(values) < 2:
                continue
            user = int(values[0])
            for raw_item in values[1:]:
                edges.add((user, int(raw_item)))
    return edges


def _build_norm_adj(edges: set[Edge], num_users: int, num_items: int) -> torch.Tensor:
    num_nodes = num_users + num_items
    if not edges:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            (num_nodes, num_nodes),
        ).coalesce()

    rows: list[int] = []
    cols: list[int] = []
    for user, item in edges:
        item_node = num_users + item
        rows.extend((user, item_node))
        cols.extend((item_node, user))

    index = torch.tensor([rows, cols], dtype=torch.long)
    value = torch.ones(len(rows), dtype=torch.float32)
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, index[0], value)
    norm_value = value * degree[index[0]].clamp_min(1).pow(-0.5) * degree[index[1]].clamp_min(1).pow(-0.5)
    return torch.sparse_coo_tensor(index, norm_value, (num_nodes, num_nodes)).coalesce()


def _build_bbp_edge_sets(
    behaviors: tuple[str, ...],
    train_edges: dict[str, set[Edge]],
) -> tuple[list[tuple[str, ...]], list[set[Edge]]]:
    names: list[tuple[str, ...]] = []
    edge_sets: list[set[Edge]] = []
    for size in range(1, len(behaviors) + 1):
        for combo in itertools.combinations(behaviors, size):
            current = set.intersection(*(train_edges[behavior] for behavior in combo))
            if current:
                names.append(combo)
                edge_sets.append(current)
    return names, edge_sets


def _build_pattern_features(
    bbp_edge_sets: list[set[Edge]],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    features = torch.zeros((num_users + num_items, len(bbp_edge_sets)), dtype=torch.float32)
    for col, edges in enumerate(bbp_edge_sets):
        for user, item in edges:
            features[user, col] += 1.0
            features[num_users + item, col] += 1.0
    row_sum = features.sum(dim=1, keepdim=True).clamp_min(1.0)
    return features / row_sum


def _edges_by_user(edges: set[Edge]) -> dict[int, set[int]]:
    by_user: dict[int, set[int]] = {}
    for user, item in edges:
        by_user.setdefault(user, set()).add(item)
    return by_user


def _all_edges_by_user(edge_sets: dict[str, set[Edge]]) -> dict[int, set[int]]:
    by_user: dict[int, set[int]] = {}
    for edges in edge_sets.values():
        for user, item in edges:
            by_user.setdefault(user, set()).add(item)
    return by_user


def _item_train_counts(edge_sets: dict[str, set[Edge]], num_items: int) -> tuple[int, ...]:
    counts = [0] * num_items
    for edges in edge_sets.values():
        for _, item in edges:
            counts[item] += 1
    return tuple(counts)

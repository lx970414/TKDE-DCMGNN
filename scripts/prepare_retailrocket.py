from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

EVENT_MAP = {
    "view": "view",
    "addtocart": "cart",
    "transaction": "buy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RetailRocket events.csv into DCMGNN behavior files."
    )
    parser.add_argument("--events", required=True, help="Path to Kaggle RetailRocket events.csv.")
    parser.add_argument("--output", default="data/Retail_Rocket_raw")
    parser.add_argument("--min-user-buy", type=int, default=1)
    parser.add_argument("--min-item-buy", type=int, default=1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_path = Path(args.events)
    output = Path(args.output)

    interactions = _read_events(events_path)
    interactions = _filter_by_target_support(
        interactions,
        min_user_buy=args.min_user_buy,
        min_item_buy=args.min_item_buy,
    )
    user_map, item_map = _make_id_maps(interactions)
    split = _split_interactions(interactions, user_map, item_map, args.test_ratio)
    _write_behavior_files(split, output)

    print(f"wrote {output}")
    for behavior in ("view", "cart", "buy"):
        for name in ("train", "test"):
            edge_count = sum(len(items) for items in split[name][behavior].values())
            user_count = len(split[name][behavior])
            print(f"{behavior} {name}: users={user_count} edges={edge_count}")


def _read_events(path: Path) -> dict[str, dict[str, list[tuple[int, str]]]]:
    interactions: dict[str, dict[str, list[tuple[int, str]]]] = {
        "view": defaultdict(list),
        "cart": defaultdict(list),
        "buy": defaultdict(list),
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            behavior = EVENT_MAP.get(row["event"])
            if behavior is None:
                continue
            user = row["visitorid"]
            item = row["itemid"]
            timestamp = int(row["timestamp"])
            interactions[behavior][user].append((timestamp, item))
    return interactions


def _filter_by_target_support(
    interactions: dict[str, dict[str, list[tuple[int, str]]]],
    min_user_buy: int,
    min_item_buy: int,
) -> dict[str, dict[str, list[tuple[int, str]]]]:
    buy_by_user = interactions["buy"]
    item_counts: dict[str, int] = defaultdict(int)
    for rows in buy_by_user.values():
        for _, item in rows:
            item_counts[item] += 1

    keep_users = {
        user for user, rows in buy_by_user.items() if len({item for _, item in rows}) >= min_user_buy
    }
    keep_items = {item for item, count in item_counts.items() if count >= min_item_buy}

    filtered: dict[str, dict[str, list[tuple[int, str]]]] = {
        "view": defaultdict(list),
        "cart": defaultdict(list),
        "buy": defaultdict(list),
    }
    for behavior, by_user in interactions.items():
        for user, rows in by_user.items():
            if user not in keep_users:
                continue
            kept_rows = [(ts, item) for ts, item in rows if item in keep_items]
            if kept_rows:
                filtered[behavior][user].extend(kept_rows)
    return filtered


def _make_id_maps(
    interactions: dict[str, dict[str, list[tuple[int, str]]]],
) -> tuple[dict[str, int], dict[str, int]]:
    users = sorted({user for by_user in interactions.values() for user in by_user})
    items = sorted({item for by_user in interactions.values() for rows in by_user.values() for _, item in rows})
    return (
        {user: index for index, user in enumerate(users)},
        {item: index for index, item in enumerate(items)},
    )


def _split_interactions(
    interactions: dict[str, dict[str, list[tuple[int, str]]]],
    user_map: dict[str, int],
    item_map: dict[str, int],
    test_ratio: float,
) -> dict[str, dict[str, dict[int, set[int]]]]:
    split: dict[str, dict[str, dict[int, set[int]]]] = {
        "train": {behavior: defaultdict(set) for behavior in EVENT_MAP.values()},
        "test": {behavior: defaultdict(set) for behavior in EVENT_MAP.values()},
    }

    for behavior, by_user in interactions.items():
        for raw_user, rows in by_user.items():
            dedup: dict[str, int] = {}
            for timestamp, raw_item in rows:
                dedup[raw_item] = max(timestamp, dedup.get(raw_item, timestamp))
            ordered_items = [
                raw_item for raw_item, _ in sorted(dedup.items(), key=lambda item_ts: item_ts[1])
            ]
            if not ordered_items:
                continue
            test_count = max(1, int(len(ordered_items) * test_ratio)) if len(ordered_items) > 1 else 0
            user = user_map[raw_user]
            train_items = ordered_items[:-test_count] if test_count else ordered_items
            test_items = ordered_items[-test_count:] if test_count else []
            for raw_item in train_items:
                split["train"][behavior][user].add(item_map[raw_item])
            for raw_item in test_items:
                split["test"][behavior][user].add(item_map[raw_item])
    return split


def _write_behavior_files(
    split: dict[str, dict[str, dict[int, set[int]]]],
    output: Path,
) -> None:
    for behavior in EVENT_MAP.values():
        behavior_dir = output / behavior
        behavior_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train", "test"):
            _write_user_items(behavior_dir / f"{name}.txt", split[name][behavior])


def _write_user_items(path: Path, user_items: dict[int, set[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user in sorted(user_items):
            items = sorted(user_items[user])
            if items:
                handle.write(" ".join([str(user), *(str(item) for item in items)]) + "\n")


if __name__ == "__main__":
    main()

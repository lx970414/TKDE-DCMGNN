from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmgnn.config import get_dataset_config
from dcmgnn.data import load_behavior_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a DCMGNN behavior dataset.")
    parser.add_argument("--dataset", required=True, choices=["tmall", "Retail_Rocket", "yelp"])
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_dataset_config(args.dataset, Path(args.data_root))
    missing = [
        config.root / behavior / split
        for behavior in config.behaviors
        for split in ("train.txt", "test.txt")
        if not (config.root / behavior / split).exists()
    ]
    if config.name == "yelp" and missing:
        print(f"dataset={config.name}")
        print(f"root={config.root}")
        print("missing required Yelp files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(2)
    dataset = load_behavior_dataset(config)
    print(f"dataset={config.name}")
    print(f"root={config.root}")
    print(f"users={dataset.num_users} items={dataset.num_items}")
    print(f"behaviors={','.join(config.behaviors)}")
    print(f"target={config.target_behavior}")
    print(f"relation_order={'->'.join(config.relation_order)}")
    print(f"bbp_count={len(dataset.bbp_names)}")
    for behavior in config.behaviors:
        train_count = len(dataset.train_edges.get(behavior, set()))
        test_count = len(dataset.test_edges.get(behavior, set()))
        print(f"{behavior}: train={train_count} test={test_count}")
    target_train = sum(len(items) for items in dataset.target_train_by_user.values())
    target_test = sum(len(items) for items in dataset.target_test_by_user.values())
    print(f"target_train_interactions={target_train}")
    print(f"target_test_interactions={target_test}")
    print(f"target_train_users={len(dataset.target_train_by_user)}")
    print(f"target_test_users={len(dataset.target_test_by_user)}")


if __name__ == "__main__":
    main()

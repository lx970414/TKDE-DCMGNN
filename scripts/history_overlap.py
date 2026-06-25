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
    parser = argparse.ArgumentParser(description="Check target test overlap with non-target training history.")
    parser.add_argument("--dataset", default="tmall", choices=["tmall", "Retail_Rocket", "yelp"])
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_behavior_dataset(get_dataset_config(args.dataset, args.data_root))
    total = 0
    overlap = 0
    users_with_overlap = 0
    for user, positives in dataset.target_test_by_user.items():
        non_target_history = set(dataset.all_train_by_user.get(user, set())) - set(
            dataset.target_train_by_user.get(user, set())
        )
        user_overlap = len(set(positives) & non_target_history)
        overlap += user_overlap
        total += len(positives)
        if user_overlap > 0:
            users_with_overlap += 1
    ratio = overlap / total if total else 0.0
    print(f"target_test_positives={total}")
    print(f"overlap_with_non_target_history={overlap}")
    print(f"overlap_ratio={ratio:.6f}")
    print(f"users_with_overlap={users_with_overlap}")


if __name__ == "__main__":
    main()

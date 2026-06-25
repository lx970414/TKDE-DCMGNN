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
    parser = argparse.ArgumentParser(description="Summarize target-only vs all-behavior seen-item masks.")
    parser.add_argument("--dataset", default="tmall", choices=["tmall", "Retail_Rocket", "yelp"])
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_behavior_dataset(get_dataset_config(args.dataset, args.data_root))
    users = list(dataset.target_test_by_user)
    target_seen = [len(dataset.target_train_by_user.get(user, set())) for user in users]
    all_seen = [len(dataset.all_train_by_user.get(user, set())) for user in users]
    extra_seen = [all_count - target_count for all_count, target_count in zip(all_seen, target_seen)]
    print(f"users={len(users)}")
    for name, values in (
        ("target_seen", target_seen),
        ("extra_seen_non_target", extra_seen),
        ("all_seen", all_seen),
    ):
        print(_describe(name, values))


def _describe(name: str, values: list[int]) -> str:
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    return (
        f"{name}: mean={mean:.2f} median={ordered[count // 2]} "
        f"p90={ordered[int(count * 0.90)]} p99={ordered[int(count * 0.99)]} max={ordered[-1]}"
    )


if __name__ == "__main__":
    main()

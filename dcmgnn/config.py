from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    behaviors: tuple[str, ...]
    target_behavior: str
    relation_order: tuple[str, ...]


def get_dataset_config(name: str, data_root: str | Path = "data") -> DatasetConfig:
    """Return the behavior order used by the paper for a supported dataset."""
    key = name.lower()
    base = Path(data_root)

    if key in {"tmall", "t-mall"}:
        return DatasetConfig(
            name="tmall",
            root=base / "tmall",
            behaviors=("view", "cart", "buy"),
            target_behavior="buy",
            relation_order=("view", "cart", "buy"),
        )

    if key in {"retail", "retail_rocket", "retail-rocket"}:
        return DatasetConfig(
            name="Retail_Rocket",
            root=base / "Retail_Rocket",
            behaviors=("view", "cart", "buy"),
            target_behavior="buy",
            relation_order=("view", "cart", "buy"),
        )

    if key in {"yelp"}:
        return DatasetConfig(
            name="yelp",
            root=base / "yelp",
            behaviors=("dislike", "neutral", "tips", "like"),
            target_behavior="like",
            relation_order=("neutral", "tips", "like"),
        )

    raise ValueError(
        f"Unsupported dataset {name!r}. Supported datasets: tmall, Retail_Rocket, yelp."
    )

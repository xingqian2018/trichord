import json
import math
import random
from pathlib import Path
from typing import Iterator, Optional

import yaml

TAXONOMY_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = "poster_topics"
DEFAULT_SEP = "|"
SUFFIXES = (".yaml", ".yml", ".json")


TOPIC_MESSAGE_TEMPLATE = "Please generate a poster with the following topic:\n{topic}"


def resolve_taxonomy_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix in SUFFIXES:
        return path
    for suffix in SUFFIXES:
        candidate = TAXONOMY_DIR / f"{name_or_path}{suffix}"
        if candidate.is_file():
            return candidate
    return TAXONOMY_DIR / f"{name_or_path}.yaml"


def list_taxonomies() -> list[str]:
    return sorted({p.stem for suffix in SUFFIXES for p in TAXONOMY_DIR.glob(f"*{suffix}")})


def split_weight(key: str, sep: str = DEFAULT_SEP) -> tuple[str, Optional[float]]:
    name, found, weight = key.rpartition(sep)
    if found and name:
        try:
            value = float(weight)
        except ValueError:
            return key, None
        if math.isfinite(value) and value >= 0:
            return name, value
    return key, None


class Taxonomy:
    """A topic tree loaded from YAML or JSON. A leaf is a key whose value is null or an empty mapping.
    Keys may carry a sampling weight as "name|weight" (e.g. t2i_arena_v1.json); then topics are drawn level by level in
    proportion to the weights, renormalized among siblings at every level (a level need not sum to 1; a leaf's probability
    is the product of its renormalized weights along the path), and the weights are stripped from the topic path.
    A weight is a finite, non-negative float after the last separator; anything else stays part of the name. A sibling
    without a weight gets the mean of its weighted siblings; a level whose weights sum to 0 is sampled uniformly.
    Without weights every leaf is equally likely."""

    def __init__(self, name_or_path: str | Path = DEFAULT_TAXONOMY, sep: str = DEFAULT_SEP):
        self.path = resolve_taxonomy_path(name_or_path)
        self.sep = sep
        text = self.path.read_text()
        self.tree: dict = (json.loads(text) if self.path.suffix == ".json" else yaml.safe_load(text)) or {}
        self.weighted = False
        found = list(self.iter_leaves(self.tree))
        self.leaves: list[tuple[str, ...]] = [leaf for leaf, _ in found]
        self.weights: list[float] = [weight for _, weight in found]
        self.topics: list[str] = [self.sep.join(leaf) for leaf in self.leaves]

    def iter_leaves(self, tree: dict, path: tuple[str, ...] = (), prob: float = 1.0) -> Iterator[tuple[tuple[str, ...], float]]:
        parsed = [(*split_weight(key, self.sep), child) for key, child in tree.items()]
        given = [weight for _, weight, _ in parsed if weight is not None]
        if given:
            self.weighted = True
        fill = sum(given) / len(given) if given else 1.0
        weights = [weight if weight is not None else fill for _, weight, _ in parsed]
        total = sum(weights)
        if total <= 0:
            weights, total = [1.0] * len(parsed), float(len(parsed))
        for (name, _, child), weight in zip(parsed, weights):
            p = prob * weight / total
            if not child:
                yield (*path, name), p
            else:
                yield from self.iter_leaves(child, (*path, name), p)

    def list_topics(self) -> list[str]:
        return list(self.topics)

    def random_get_one_topic(self, seed: Optional[int] = None) -> str:
        rng = random.Random(seed)
        if self.weighted:
            return rng.choices(self.topics, weights=self.weights)[0]
        return rng.choice(self.topics)

    def random_get_topics(self, num: int, seed: Optional[int] = None) -> list[str]:
        rng = random.Random(seed)
        if self.weighted:
            return rng.choices(self.topics, weights=self.weights, k=num)
        if num > len(self.topics):
            return [rng.choice(self.topics) for _ in range(num)]
        return rng.sample(self.topics, num)

    def split_topic(self, topic: str) -> list[str]:
        return topic.split(self.sep)

    def __len__(self) -> int:
        return len(self.topics)

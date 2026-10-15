import random
from pathlib import Path
from typing import Iterator, Optional

import yaml

TAXONOMY_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = "poster_topics"
DEFAULT_SEP = "|"


TOPIC_MESSAGE_TEMPLATE = "Please generate a poster with the following topic:\n{topic}"


def resolve_taxonomy_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    return TAXONOMY_DIR / f"{name_or_path}.yaml"


def list_taxonomies() -> list[str]:
    return sorted(p.stem for p in TAXONOMY_DIR.glob("*.yaml"))


class Taxonomy:
    def __init__(self, name_or_path: str | Path = DEFAULT_TAXONOMY, sep: str = DEFAULT_SEP):
        self.path = resolve_taxonomy_path(name_or_path)
        self.sep = sep
        self.tree: dict = yaml.safe_load(self.path.read_text()) or {}
        self.leaves: list[tuple[str, ...]] = list(self.iter_leaves(self.tree))
        self.topics: list[str] = [self.sep.join(leaf) for leaf in self.leaves]

    @staticmethod
    def iter_leaves(tree: dict, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
        for key, child in tree.items():
            if child is None:
                yield (*path, key)
            else:
                yield from Taxonomy.iter_leaves(child, (*path, key))

    def list_topics(self) -> list[str]:
        return list(self.topics)

    def random_get_one_topic(self, seed: Optional[int] = None) -> str:
        return random.Random(seed).choice(self.topics)

    def random_get_topics(self, num: int, seed: Optional[int] = None) -> list[str]:
        rng = random.Random(seed)
        if num > len(self.topics):
            return [rng.choice(self.topics) for _ in range(num)]
        return rng.sample(self.topics, num)

    def split_topic(self, topic: str) -> list[str]:
        return topic.split(self.sep)

    def __len__(self) -> int:
        return len(self.topics)

"""Resolve legacy dataset aliases; preserve their exact resolved revisions.

Opt-in client-only shim. No task prompts, examples, sampling or scoring change.
"""
import functools
import sys

import datasets

ALIASES = {
    'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866'),
    'mbpp': ('google-research-datasets/mbpp', '4bb6404fdc6cacfda99d4ac4205087b89d32030c'),
    'hellaswag': ('Rowan/hellaswag', '218ec52e09a7e7462a5400043bb9a69a41d06b76'),
}
original_load_dataset = datasets.load_dataset


@functools.wraps(original_load_dataset)
def canonical_load_dataset(path, *args, **kwargs):
    if path in ALIASES:
        canonical, revision = ALIASES[path]
        if kwargs.get('revision') is None:
            kwargs['revision'] = revision
        print(f'Dataset alias compatibility: {path} -> {canonical}@{kwargs["revision"]}',
              file=sys.stderr, flush=True)
        path = canonical
    return original_load_dataset(path, *args, **kwargs)


datasets.load_dataset = canonical_load_dataset


# nemo6 external-reference runs (ours, not Boxiang's): stop<=4 for vLLM v0.31, LCB execution memory caps - see nemo6_patches.py
import nemo6_patches  # noqa: E402,F401

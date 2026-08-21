import functools
import re
import time
from itertools import chain, combinations


def list_of_sets_repr(ls: list[set | frozenset]) -> (str | None):
    if len(ls) == 0:
        return 'None'
    return str(', '.join([set_repr(s) for s in ls]))


def set_repr(s: set | frozenset) -> (str | None):
    return '{' + ', '.join(sorted(s)) + '}'


def list_repr(l: list) -> (str | None):
    return '[' + ', '.join(l) + ']'


def parse_set_repr(s: str) -> set:
    content = s.strip('{}')
    if not content:
        return set()
    return set(content.split(', '))


def safe_str(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', s)


def format_num(n_samples: int) -> str:
    if n_samples >= 1000000:
        return f'{n_samples // 1000000}M'
    elif n_samples >= 1000:
        return f'{n_samples // 1000}K'
    else:
        return str(n_samples)


def find_powerset(iterable) -> list[frozenset]:
    s = list(iterable)
    p = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    powerset = [frozenset(t) for t in p if t]
    return sorted(powerset, key=len)


def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed


def dataset_label(dataset_class) -> str:
    extras = []

    # Only do it if it's a factory (partial)
    if isinstance(dataset_class, functools.partial):
        name = getattr(dataset_class, "name", None)
        rank = getattr(dataset_class, "rank", None)
        init = getattr(dataset_class, "init", None)
        sample = getattr(dataset_class, "sample", None)
        base = getattr(dataset_class, "__name__", "PQRS")

        # `name`/`sample` are set by the Real - Default factory; synthetic
        # partials only set rank/init, so those stay unaffected.
        if name is not None: extras.append(f"{name}")
        if rank is not None: extras.append(f"rank_{rank}")
        if init is not None: extras.append(f"init_{init}")
        if sample is not None: extras.append(f"s{sample}")

    return extras

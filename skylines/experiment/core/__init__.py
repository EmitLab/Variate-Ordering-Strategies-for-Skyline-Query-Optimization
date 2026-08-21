from collections.abc import Callable, Iterable, Iterator
from typing import Any

from dask.distributed import Client, as_completed


def windowed_map(
    client: Client,
    fn: Callable,
    configs: Iterable[Any],
    window: int,
) -> Iterator[Any]:
    """Submit at most `window` futures at a time; yield results in completion order.

    Throttles on the submission side so the Dask scheduler never sees more than
    `window` in-flight futures of `fn`. Avoids the eager-distribution deadlock
    of in-task semaphores: when `len(configs) > n_workers`, an in-task
    Semaphore.acquire() leaves all assigned tasks blocked while holding worker
    slots, starving the inner tasks they would have launched.
    """
    pending = list(configs)
    cursor = 0

    in_flight = []
    while cursor < len(pending) and len(in_flight) < window:
        in_flight.append(client.submit(fn, pending[cursor]))
        cursor += 1

    if not in_flight:
        return

    seq = as_completed(in_flight)
    for fut in seq:
        yield fut.result()
        if cursor < len(pending):
            new_fut = client.submit(fn, pending[cursor])
            cursor += 1
            seq.add(new_fut)

"""
Real - Default experiment factory.

Implements the "Default experiment set" from `real_world experiment_plan.md`:

  For each real-world dataset, enumerate every ``pref_size``-variate column
  subset, bucket it by the *effective rank* of its correlation submatrix
  (smallest k whose top-k eigenvalues carry >= 95% of the variance — the
  same metric used in analysis_notebooks/preference_rank_analysis.ipynb),
  then for each rank ``r`` in ``ranks`` take ``count_r = ceil(k% · |pool_r|)``
  preference sets, sampled **without replacement**, where the ``k_i``-th
  pick (``k_i`` in ``[0, count_r)``) uses ``random_state = k_i``.

Notes:
  * ``k`` in skylines/__init__.py is a **percentage** (``k = 10`` ⇒ 10%),
    not an absolute count. There is no ``init`` dimension.
  * (dataset, rank) cells with an empty pool are skipped.
  * ``1 <= count_r <= |pool_r|`` for any non-empty pool (``ceil``), so
    sampling without replacement is always valid.
  * Selection is deterministic: ``pool_r`` is sorted, then the ``k_i``-th
    element is drawn from the remaining pool with ``random.Random(k_i)``.

Rank is recomputed in-code from each dataset's raw CSV so the experiment is
self-contained and does not depend on the analysis-notebook CSV artifact.
"""

import functools
import itertools
import math
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from skylines.dataset import Dominance
from skylines.dataset.real.real import RealDataset

DATA_DIR = Path(__file__).parent / 'data'

# Variance threshold for the effective-rank metric. Mirrors VAR_THRESHOLD in
# analysis_notebooks/preference_rank_analysis.ipynb — keep the two in sync.
VAR_THRESHOLD = 0.95

# Non-ordinal / categorical columns excluded from the candidate pool before
# enumerating preference subsets (real_world experiment_plan.md §2). Keyed by
# dataset dir name. Missing columns are ignored, so listing a non-numeric col
# already dropped by to_numeric is harmless. Excluded in code only — the raw
# CSV files are never modified. Keep in sync with EXCLUDE in
# analysis_notebooks/preference_rank_analysis.ipynb.
_EXCLUDE: dict[str, list[str]] = {
    'abalone':                                ['Sex'],
    'adult':                                  ['workclass', 'education', 'education-num',
                                               'marital-status', 'occupation', 'relationship',
                                               'race', 'sex', 'native-country'],
    'autompg':                                ['car_name'],
    'concrete_compressive_strength':          [],
    'drybean':                                ['Class'],
    'hong_kong_weather':                      [],
    'individual_household_power_consumption': ['Date'],
    'seoul_bike_demand':                      ['Date', 'Seasons', 'Functioning Day', 'Holiday'],
    'wine_quality_white':                     [],
}


class GenericRealDataset(RealDataset):
    """A RealDataset over an arbitrary preference-column subset.

    The existing per-dataset classes (e.g. ``Abalone_5var_mixed``) hardcode
    their preference columns; this generic variant takes the subset as a
    constructor argument so the Real - Default factory can build one dataset
    per sampled combo. ``control`` is empty (only the preference columns are
    loaded) — matching the synthetic VOS convention.
    """

    def __init__(self, dominance=None, size=None, seed=42, *,
                 file_name: str, preference: list[str]):
        preference = list(preference)
        super().__init__(
            file_name=file_name,
            control=[],
            preference=preference,
            effect={p: {} for p in preference},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in preference},
            size=size,
            seed=seed,
            provided=False,
        )


def _load_numeric(csv_path: Path, exclude: list[str] | None = None) -> pd.DataFrame:
    """Numeric, non-constant columns of a raw dataset CSV.

    Mirrors ``load_numeric`` in preference_rank_analysis.ipynb so the candidate
    column pool (and therefore the rank buckets) match the analysis. ``exclude``
    drops the non-ordinal/categorical columns (§2) up front — in memory only,
    the CSV on disk is untouched.
    """
    df = pd.read_csv(csv_path)
    if exclude:
        df = df.drop(columns=exclude, errors='ignore')   # categorical cols (§2)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')                # drop non-numeric cols
    df = df.loc[:, df.std(numeric_only=True) > 0]    # drop constant cols
    return df.dropna()


def _effective_rank(corr: np.ndarray, threshold: float = VAR_THRESHOLD) -> int:
    """Smallest k s.t. cumulative top-k eigenvalues / total >= threshold."""
    eigvals = np.linalg.eigvalsh(corr)            # ascending, real (symmetric)
    eigvals = np.clip(eigvals[::-1], 0, None)     # descending, drop tiny negs
    total = eigvals.sum()
    if total == 0:
        return 0
    cum = np.cumsum(eigvals) / total
    return int(np.searchsorted(cum, threshold) + 1)


def get_vos_real_default_datasets(
    dataset_dir: str,
    pref_size: int,
    ranks: Iterable[int],
    k_pct: float,
) -> list[tuple[functools.partial, None]]:
    """Sampled rank-bucketed preference subsets for one real dataset.

    ``k_pct`` is a percentage (e.g. ``10`` ⇒ 10%): per rank ``r`` with a
    non-empty pool, ``count_r = ceil(k_pct/100 · |pool_r|)`` subsets are
    sampled without replacement.

    Returns a list of ``(dataset_class, None)`` pairs — the same shape as
    ``get_vos_synth_datasets`` — one entry per sampled preference set. Each
    ``dataset_class`` is a ``functools.partial`` carrying ``name``, ``rank``
    and ``sample`` (= ``k_i``) attributes so ``dataset_label`` produces a
    unique workbook name.
    """
    ranks = list(ranks)
    csv = DATA_DIR / dataset_dir / f'{dataset_dir}.csv'
    if not csv.exists():
        return []

    df = _load_numeric(csv, _EXCLUDE.get(dataset_dir, []))   # drop categorical (§2)
    cols = list(df.columns)
    if len(cols) < pref_size:
        return []

    corr_full = df.corr().to_numpy()
    col_idx = {c: i for i, c in enumerate(cols)}

    # Bucket every pref_size-combo by the effective rank of its corr submatrix.
    pools: dict[int, list[tuple[str, ...]]] = {r: [] for r in ranks}
    for combo in itertools.combinations(cols, pref_size):
        idx = [col_idx[c] for c in combo]
        sub = corr_full[np.ix_(idx, idx)]
        r = _effective_rank(sub)
        if r in pools:
            pools[r].append(combo)

    datasets: list[tuple[functools.partial, None]] = []
    for r in ranks:
        pool = sorted(pools[r])          # deterministic order before sampling
        if not pool:
            continue                     # skip empty (dataset, rank) cell
        # count_r = ceil(k% of |pool_r|), clamped to the pool size.
        count_r = min(math.ceil(k_pct / 100.0 * len(pool)), len(pool))
        remaining = list(pool)
        for k_i in range(count_r):
            rng = random.Random(k_i)     # random_state = k_i (per the plan)
            combo = remaining.pop(rng.randrange(len(remaining)))  # no replacement
            ds = functools.partial(
                GenericRealDataset,
                file_name=dataset_dir,
                preference=list(combo),
            )
            ds.name = dataset_dir
            ds.rank = r
            ds.sample = k_i
            ds.__name__ = dataset_dir
            datasets.append((ds, None))

    return datasets

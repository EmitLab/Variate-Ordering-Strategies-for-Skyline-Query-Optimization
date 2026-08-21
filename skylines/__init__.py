import numpy as np

""" SIMD width for dominance kernel: 'scalar' | 'sse' | 'avx' """
simd = 'scalar'

""" Data dtype: 'f32' or 'f64' """
dtype = 'f64'

# np.dtype derived from the string knob above. Use this everywhere in code.
data_dtype = np.float64 if dtype == 'f64' else np.float32

""" Number of Dask workers """
n_workers = 75
ensemble_limit = 5
seed_limit = 5

# """ 200K Number of Dask workers """
# n_workers = 48
# ensemble_limit = 4
# seed_limit = 5

""" Number of times to sample for cov matrices """
n_inits = 5

""" Real - Default: PERCENT of each rank pool to sample (k = 10 -> 10%);
    per rank r take ceil(k/100 * |pool_r|) subsets, no replacement """
k = 10

""" Number of tuples """
n_samples = 100_000

""" Number of runs for each experiment (to ensure stability) """
n_runs = 5

""" Lexicographic sort based on variate order. Either a single bool — broadcast
to every algorithm — or a per-algorithm dict keyed by algorithm label
('BNL', 'SFS', 'SaLSa', 'D&C', 'BBS'); omitted keys default to False. A dict
lets one run mix lexsort and plain algorithms, e.g.
    lex_sort = {'SFS': True, 'SaLSa': True}
runs SFS/SaLSa with the lexsort presort and BNL/D&C/BBS plain. """
lex_sort = True

# Canonical algorithm labels recognised in a per-algorithm `lex_sort` dict.
# Must match the labels used by the experiment driver (Experiment.execute).
lex_sort_algorithms = ('BNL', 'SFS', 'SaLSa', 'D&C', 'BBS')


def resolve_lex_sort(cfg=None):
    """Normalize the `lex_sort` knob into a {algorithm: bool} map.

    `cfg` defaults to the live module-level `lex_sort`. A plain bool is
    broadcast to every algorithm in `lex_sort_algorithms`; a dict is validated
    against that set, with omitted keys defaulting to False. Raises ValueError
    on an unknown algorithm key and TypeError on a non-bool / non-dict value.
    """
    if cfg is None:
        cfg = lex_sort
    if isinstance(cfg, bool):
        return {algo: cfg for algo in lex_sort_algorithms}
    if isinstance(cfg, dict):
        unknown = set(cfg) - set(lex_sort_algorithms)
        if unknown:
            raise ValueError(
                f"invalid lex_sort algorithm key(s) {sorted(unknown)}; "
                f"expected a subset of {list(lex_sort_algorithms)}")
        return {algo: bool(cfg.get(algo, False)) for algo in lex_sort_algorithms}
    raise TypeError(
        f"lex_sort must be a bool or dict[str, bool], got {type(cfg).__name__}")

""" Skyline algorithms to run for each experiment. Either the string 'all' —
runs every algorithm in `lex_sort_algorithms` — or a list/tuple/set of algorithm
labels ('BNL', 'SFS', 'SaLSa', 'D&C', 'BBS') to run only a subset, e.g.
    run_algorithms = ['SFS', 'SaLSa']
runs only SFS and SaLSa and leaves the other three out of the results entirely.
Algorithms always execute (and appear in result columns) in the canonical
`lex_sort_algorithms` order regardless of the order listed here. Ignored by the
monotonic_experiment path, which always runs SFS + SaLSa. """
# run_algorithms = 'all'
run_algorithms = ['SFS', 'SaLSa']

def resolve_run_algorithms(cfg=None):
    """Normalize the `run_algorithms` knob into an ordered list of algorithm labels.

    `cfg` defaults to the live module-level `run_algorithms`. The string 'all'
    (case-insensitive) expands to every label in `lex_sort_algorithms`. A
    list/tuple/set of labels is validated against that set and returned in
    canonical `lex_sort_algorithms` order with duplicates removed. Raises
    ValueError on an unknown label or an empty selection, and TypeError on any
    other value.
    """
    if cfg is None:
        cfg = run_algorithms
    if isinstance(cfg, str):
        if cfg.lower() == 'all':
            return list(lex_sort_algorithms)
        raise ValueError(
            f"invalid run_algorithms string {cfg!r}; "
            f"the only accepted string is 'all'")
    if isinstance(cfg, (list, tuple, set)):
        requested = set(cfg)
        unknown = requested - set(lex_sort_algorithms)
        if unknown:
            raise ValueError(
                f"invalid run_algorithms label(s) {sorted(unknown)}; "
                f"expected a subset of {list(lex_sort_algorithms)}")
        selected = [algo for algo in lex_sort_algorithms if algo in requested]
        if not selected:
            raise ValueError(
                "run_algorithms selects no algorithms; set it to 'all' or a "
                f"non-empty subset of {list(lex_sort_algorithms)}")
        return selected
    raise TypeError(
        f"run_algorithms must be 'all' or a list/tuple/set of labels, "
        f"got {type(cfg).__name__}")

""" Variate-ordering strategy used as the lex-permutation key for every VO
strategy. The skyline dominance scan still walks columns in the strategy's
own VO; only the lex permutation key is set by this. Allowed values:
  "XCF" — algorithms.max_corr  (force max-correlation ordering)
  "NCF" — algorithms.min_corr  (force min-correlation ordering)
  None  — matched-VO: each strategy lex-sorts by its own VO column order
Affects BNL/SFS/SaLSa/DnC lexsort variants. BBS untouched (its R-tree heap
priority is structurally tied to the column ordering). Only takes effect when
`lex_sort` is also True. (Replaces the old boolean `lex_sort_xcf_key`:
True → "XCF", False → None.) """
lex_sort_key = "NCF"

""" Candidate-window scan direction for the BNL/SFS *lexsort* variants.
True  = reverse scan (newest→oldest skyline buffer entry; default, preserves
prior behavior). False = forward scan (oldest→newest). Only affects BNL/SFS
lexsort kernels and only when `lex_sort` is True. DnC/SaLSa/BBS lexsort have
no reverse-scan analog and are unaffected. """
lex_reverse_scan = True

""" Monotonic-function analysis experiment. When True, the experiment driver
runs ONLY SFS and SaLSa via the monosort presort path (BNL, D&C, BBS are
skipped), one run per configured monotone function. The lexsort variants are
bypassed regardless of `lex_sort` since they carry no monotone presort. """
monotonic_experiment = False

""" Monotone presort function(s) per algorithm for the monosort path. Each is a
list; the experiment runs that algorithm once per list entry, so a list with
both entries sweeps both functions in a single run. Allowed entries:
  'sum'     — descending row-sum
  'max_sum' — descending (max, sum) lexicographic
Only consulted when `monotonic_experiment` is True. """
sfs_monotonic = ['sum']
salsa_monotonic = ['max_sum']

""" Experiment type to run (enable only one line below) """
experiment_type = 'Synthetic - P Sweep'


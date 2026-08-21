from itertools import permutations
import logging
import pandas as pd
from bunch_py3 import Bunch
from dask.distributed import get_client

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from skylines.common.display.data import display_info, plot_correlation, plot_correlation_preferences, plot_dataset_graph
from skylines.common.display.result import display_results
from skylines.common.excel import Excel
from skylines.common.state import State, set_state, get_state
from skylines.common.utils import find_powerset, list_repr, time_function
from skylines.dataset import Dataset
from skylines.heuristic import *
from skylines.heuristic import algorithms
from skylines.skyline import *
from skylines import simd as _simd, lex_sort_key, resolve_lex_sort, resolve_run_algorithms
from skylines import monotonic_experiment, sfs_monotonic, salsa_monotonic

# SIMD chunk metric only meaningful when a SIMD width is selected.
_simd_active = _simd != 'scalar'
from skylines import n_samples

import warnings
warnings.filterwarnings('ignore')


# Phase 1 software-derived metrics. Order matters: becomes the MultiIndex
# level-1 label order in results/{original,raw}/simd_derived CSVs.
# See skylines/skyline/skyline.py for the @property definitions.
PHASE1_METRICS = [
    'lane_utilization',
    'early_exit_rate',
    'scalar_tail_fraction',
    'simd_amortization',
    'dom_check_throughput',
    'sub_compare_throughput',
    'simd_chunk_throughput',
]


STRATEGY_AGGREGATES = ['Best', 'Worst', 'Average', 'Median']

# Above this dim, factorial fan-out of variate permutations and pairwise
# heuristics (min_corr_pairwise) blow up. Restrict to the three lightweight
# correlation heuristics: NCF (NC_best), NACF (ANC_best), XCF (XC_best).
HIGH_DIM_THRESHOLD = 10
HIGH_DIM_STRATEGIES = ('NC_best', 'ANC_best', 'XC_best')


def get_strategy_orders(dataset: Dataset) -> list[tuple[str, list[str]]]:
    """Named variate orderings — every heuristic kept, even when two
    heuristics resolve to the same order on a given dataset. Identical
    orders are retained here and may therefore be executed more than once.

    For datasets with more than HIGH_DIM_THRESHOLD preferences, only the
    three correlation heuristics (NCF/NACF/XCF) are used — pairwise and
    permutation-based heuristics scale factorially and are skipped."""
    if len(dataset.preference) > HIGH_DIM_THRESHOLD:
        return [
            ('NC_best', algorithms.min_corr(dataset)),
            ('ANC_best', algorithms.min_abs_corr(dataset)),
            ('XC_best', algorithms.max_corr(dataset)),
        ]
    return [
        ('NC_best', algorithms.min_corr(dataset)),
        ('ANC_best', algorithms.min_abs_corr(dataset)),
        ('XC_best', algorithms.max_corr(dataset)),
        ('NC_pairwise_best', algorithms.min_corr_pairwise(dataset)),
        ('INCF_best', algorithms.incremental_min_corr_first(dataset)),
        ('IXCL_best', algorithms.incremental_max_corr_last(dataset)),
    ]


def get_strategies(dataset: Dataset) -> list[tuple[str, list[str] | None]]:
    """Full strategy list for result post-processing.

    Aggregates (Best/Worst/Average/Median) are markers with order=None;
    they trigger the special branches in `strategies()` that reduce across
    the per-order results. Heuristic orderings come after, each producing
    a concrete per-order row.
    """
    return [(name, None) for name in STRATEGY_AGGREGATES] + get_strategy_orders(dataset)


def load_original(args):
    try:
        original_results = dict()

        in_file = get_state().get_file('results/original/dominance_checks', f'{args.excel_book}.csv')
        original_results['dominance_checks'] = pd.read_csv(in_file, index_col=0)

        in_file = get_state().get_file('results/original/variate_comparisons', f'{args.excel_book}.csv')
        original_results['variate_comparisons'] = pd.read_csv(in_file, index_col=0)

        try:
            in_file = get_state().get_file('results/original/simd_chunks', f'{args.excel_book}.csv')
            original_results['simd_chunks'] = pd.read_csv(in_file, index_col=0)
        except Exception:
            if _simd_active:
                return None
            original_results['simd_chunks'] = pd.DataFrame(0,
                                                           index=original_results['variate_comparisons'].index,
                                                           columns=original_results['variate_comparisons'].columns)
            original_results['simd_chunks'].index.name = original_results['variate_comparisons'].index.name

        in_file = get_state().get_file('results/original/timing', f'{args.excel_book}.csv')
        original_results['timing'] = pd.read_csv(in_file, index_col=0)

        in_file = get_state().get_file('results/original/parallel_time', f'{args.excel_book}.csv')
        original_results['parallel_time'] = pd.read_csv(in_file, index_col=0)

        try:
            in_file = get_state().get_file('results/original/compute_wall', f'{args.excel_book}.csv')
            original_results['compute_wall'] = pd.read_csv(in_file, index_col=0)

            in_file = get_state().get_file('results/original/python_overhead', f'{args.excel_book}.csv')
            original_results['python_overhead'] = pd.read_csv(in_file, index_col=0)
        except Exception:
            zero = pd.DataFrame(0.0,
                                index=original_results['timing'].index,
                                columns=original_results['timing'].columns)
            zero.index.name = original_results['timing'].index.name
            original_results['compute_wall'] = zero.copy()
            original_results['python_overhead'] = zero.copy()

        try:
            in_file = get_state().get_file('results/original/sort_time', f'{args.excel_book}.csv')
            original_results['sort_time'] = pd.read_csv(in_file, index_col=0)
        except Exception:
            zero = pd.DataFrame(0.0,
                                index=original_results['timing'].index,
                                columns=original_results['timing'].columns)
            zero.index.name = original_results['timing'].index.name
            original_results['sort_time'] = zero.copy()

        try:
            in_file = get_state().get_file('results/original/normalize_time', f'{args.excel_book}.csv')
            original_results['normalize_time'] = pd.read_csv(in_file, index_col=0)
        except Exception:
            zero = pd.DataFrame(0.0,
                                index=original_results['timing'].index,
                                columns=original_results['timing'].columns)
            zero.index.name = original_results['timing'].index.name
            original_results['normalize_time'] = zero.copy()

        try:
            in_file = get_state().get_file('results/original/index_time', f'{args.excel_book}.csv')
            original_results['index_time'] = pd.read_csv(in_file, index_col=0)
        except Exception:
            zero = pd.DataFrame(0.0,
                                index=original_results['timing'].index,
                                columns=original_results['timing'].columns)
            zero.index.name = original_results['timing'].index.name
            original_results['index_time'] = zero.copy()

        try:
            in_file = get_state().get_file('results/original/simd_derived', f'{args.excel_book}.csv')
            original_results['simd_derived'] = pd.read_csv(in_file, index_col=[0, 1])
        except Exception:
            clusters = original_results['timing'].index
            index = pd.MultiIndex.from_product(
                [clusters, PHASE1_METRICS], names=['Cluster', 'Metric']
            )
            original_results['simd_derived'] = pd.DataFrame(
                0.0, index=index, columns=original_results['timing'].columns
            )

        return original_results
    
    except Exception:
        return None
    

def agg_original(args):
    # run sub-experiment for each variate order
    raw_results: dict = {'dominance_checks': None, 'variate_comparisons': None, 'simd_chunks': None, 'vcheck_frequency': None, 'timing': None, 'parallel_time': None, 'compute_wall': None, 'python_overhead': None, 'sort_time': None, 'normalize_time': None, 'index_time': None, 'simd_derived': None}
    
    for result in args.results:
        order = result.order
        cluster_result = result.result

        # add cluster results to all results
        if raw_results['dominance_checks'] is None:
            raw_results['dominance_checks'] = pd.DataFrame(columns=cluster_result['dominance_checks']['Model'].values)
            raw_results['dominance_checks'].index.name = 'Cluster'
        raw_results['dominance_checks'].loc[list_repr(order)] = cluster_result['dominance_checks']['Dominance Checks'].values

        if raw_results['variate_comparisons'] is None:
            raw_results['variate_comparisons'] = pd.DataFrame(columns=cluster_result['variate_comparisons']['Model'].values)
            raw_results['variate_comparisons'].index.name = 'Cluster'
        raw_results['variate_comparisons'].loc[list_repr(order)] = cluster_result['variate_comparisons']['Variate Comparisons'].values

        if raw_results['simd_chunks'] is None:
            raw_results['simd_chunks'] = pd.DataFrame(columns=cluster_result['simd_chunks']['Model'].values)
            raw_results['simd_chunks'].index.name = 'Cluster'
        raw_results['simd_chunks'].loc[list_repr(order)] = cluster_result['simd_chunks']['SIMD Chunks Compared'].values

        if raw_results['vcheck_frequency'] is None:
            index = pd.MultiIndex.from_tuples([], names=['Cluster', 'No. of vchecks'])
            raw_results['vcheck_frequency'] = pd.DataFrame(columns=cluster_result['vcheck_frequency']['Model'].values, index=index)
        for i in range(1, cluster_result['vcheck_frequency'].shape[1]):
            raw_results['vcheck_frequency'].loc[(list_repr(order), i), :] = cluster_result['vcheck_frequency'][i].values
        
        if raw_results['timing'] is None:
            raw_results['timing'] = pd.DataFrame(columns=cluster_result['timing']['Model'].values)
            raw_results['timing'].index.name = 'Cluster'
        raw_results['timing'].loc[list_repr(order)] = cluster_result['timing']['Time (seconds)'].values

        if raw_results['parallel_time'] is None:
            raw_results['parallel_time'] = pd.DataFrame(columns=cluster_result['parallel_time']['Model'].values)
            raw_results['parallel_time'].index.name = 'Cluster'
        raw_results['parallel_time'].loc[list_repr(order)] = cluster_result['parallel_time']['Time (seconds)'].values

        if raw_results['compute_wall'] is None:
            raw_results['compute_wall'] = pd.DataFrame(columns=cluster_result['compute_wall']['Model'].values)
            raw_results['compute_wall'].index.name = 'Cluster'
        raw_results['compute_wall'].loc[list_repr(order)] = cluster_result['compute_wall']['Time (seconds)'].values

        if raw_results['python_overhead'] is None:
            raw_results['python_overhead'] = pd.DataFrame(columns=cluster_result['python_overhead']['Model'].values)
            raw_results['python_overhead'].index.name = 'Cluster'
        raw_results['python_overhead'].loc[list_repr(order)] = cluster_result['python_overhead']['Time (seconds)'].values

        if raw_results['sort_time'] is None:
            raw_results['sort_time'] = pd.DataFrame(columns=cluster_result['sort_time']['Model'].values)
            raw_results['sort_time'].index.name = 'Cluster'
        raw_results['sort_time'].loc[list_repr(order)] = cluster_result['sort_time']['Time (seconds)'].values

        if raw_results['normalize_time'] is None:
            raw_results['normalize_time'] = pd.DataFrame(columns=cluster_result['normalize_time']['Model'].values)
            raw_results['normalize_time'].index.name = 'Cluster'
        raw_results['normalize_time'].loc[list_repr(order)] = cluster_result['normalize_time']['Time (seconds)'].values

        if raw_results['index_time'] is None:
            raw_results['index_time'] = pd.DataFrame(columns=cluster_result['index_time']['Model'].values)
            raw_results['index_time'].index.name = 'Cluster'
        raw_results['index_time'].loc[list_repr(order)] = cluster_result['index_time']['Time (seconds)'].values

        if raw_results['simd_derived'] is None:
            index = pd.MultiIndex.from_tuples([], names=['Cluster', 'Metric'])
            raw_results['simd_derived'] = pd.DataFrame(columns=cluster_result['simd_derived']['Model'].values, index=index)
        cluster_metrics = cluster_result['simd_derived'].set_index('Model')
        for metric in PHASE1_METRICS:
            raw_results['simd_derived'].loc[(list_repr(order), metric), :] = cluster_metrics[metric].reindex(raw_results['simd_derived'].columns).values

    return raw_results


def save_original(args):
    # store the expanded raw/original results
    out_file = get_state().get_file('results/original/dominance_checks', f'{args.excel_book}.csv')
    args.original_results['dominance_checks'].to_csv(out_file)

    out_file = get_state().get_file('results/original/variate_comparisons', f'{args.excel_book}.csv')
    args.original_results['variate_comparisons'].to_csv(out_file)

    out_file = get_state().get_file('results/original/simd_chunks', f'{args.excel_book}.csv')
    args.original_results['simd_chunks'].to_csv(out_file)

    out_file = get_state().get_file('results/original/vcheck_frequency', f'{args.excel_book}.csv')
    args.original_results['vcheck_frequency'].to_csv(out_file)

    out_file = get_state().get_file('results/original/timing', f'{args.excel_book}.csv')
    args.original_results['timing'].to_csv(out_file)

    out_file = get_state().get_file('results/original/parallel_time', f'{args.excel_book}.csv')
    args.original_results['parallel_time'].to_csv(out_file)

    out_file = get_state().get_file('results/original/compute_wall', f'{args.excel_book}.csv')
    args.original_results['compute_wall'].to_csv(out_file)

    out_file = get_state().get_file('results/original/python_overhead', f'{args.excel_book}.csv')
    args.original_results['python_overhead'].to_csv(out_file)

    out_file = get_state().get_file('results/original/sort_time', f'{args.excel_book}.csv')
    args.original_results['sort_time'].to_csv(out_file)

    out_file = get_state().get_file('results/original/normalize_time', f'{args.excel_book}.csv')
    args.original_results['normalize_time'].to_csv(out_file)

    out_file = get_state().get_file('results/original/index_time', f'{args.excel_book}.csv')
    args.original_results['index_time'].to_csv(out_file)

    out_file = get_state().get_file('results/original/simd_derived', f'{args.excel_book}.csv')
    args.original_results['simd_derived'].to_csv(out_file)


def strategies(args):
    # rearrange results
    raw_results: dict = {'dominance_checks': args.original_results['dominance_checks'].iloc[0:0],
                        'variate_comparisons': args.original_results['variate_comparisons'].iloc[0:0],
                        'simd_chunks': args.original_results['simd_chunks'].iloc[0:0],
                        'vcheck_frequency': args.original_results['vcheck_frequency'].iloc[0:0],
                        'timing': args.original_results['timing'].iloc[0:0],
                        'parallel_time': args.original_results['parallel_time'].iloc[0:0],
                        'compute_wall': args.original_results['compute_wall'].iloc[0:0],
                        'python_overhead': args.original_results['python_overhead'].iloc[0:0],
                        'sort_time': args.original_results['sort_time'].iloc[0:0],
                        'normalize_time': args.original_results['normalize_time'].iloc[0:0],
                        'index_time': args.original_results['index_time'].iloc[0:0],
                        'simd_derived': args.original_results['simd_derived'].iloc[0:0]}

    heuristics = get_strategies(args.dataset)

    for name, order in heuristics:
        if name == 'Best':
            min_dominance_idx = args.original_results['dominance_checks'].idxmin(axis=0)
            raw_results['dominance_checks'].loc[name] = pd.Series({col: args.original_results['dominance_checks'].at[row, col] for col, row in min_dominance_idx.items()})

            min_variate_idx = args.original_results['variate_comparisons'].idxmin(axis=0)
            raw_results['variate_comparisons'].loc[name] = pd.Series({col: args.original_results['variate_comparisons'].at[row, col] for col, row in min_variate_idx.items()})

            min_simd_idx = args.original_results['simd_chunks'].idxmin(axis=0)
            raw_results['simd_chunks'].loc[name] = pd.Series({col: args.original_results['simd_chunks'].at[row, col] for col, row in min_simd_idx.items()})

            for col, row in min_variate_idx.items():
                for i in range(1, len(args.dataset.preference) + 1):
                    raw_results['vcheck_frequency'].at[(name, i), col] = args.original_results['vcheck_frequency'].at[(row, i), col]

            min_timing_idx = args.original_results['timing'].idxmin(axis=0)
            raw_results['timing'].loc[name] = pd.Series({col: args.original_results['timing'].at[row, col] for col, row in min_timing_idx.items()})

            min_parallel_idx = args.original_results['parallel_time'].idxmin(axis=0)
            raw_results['parallel_time'].loc[name] = pd.Series({col: args.original_results['parallel_time'].at[row, col] for col, row in min_parallel_idx.items()})

            min_compute_wall_idx = args.original_results['compute_wall'].idxmin(axis=0)
            raw_results['compute_wall'].loc[name] = pd.Series({col: args.original_results['compute_wall'].at[row, col] for col, row in min_compute_wall_idx.items()})

            min_python_overhead_idx = args.original_results['python_overhead'].idxmin(axis=0)
            raw_results['python_overhead'].loc[name] = pd.Series({col: args.original_results['python_overhead'].at[row, col] for col, row in min_python_overhead_idx.items()})

            min_sort_time_idx = args.original_results['sort_time'].idxmin(axis=0)
            raw_results['sort_time'].loc[name] = pd.Series({col: args.original_results['sort_time'].at[row, col] for col, row in min_sort_time_idx.items()})

            min_normalize_time_idx = args.original_results['normalize_time'].idxmin(axis=0)
            raw_results['normalize_time'].loc[name] = pd.Series({col: args.original_results['normalize_time'].at[row, col] for col, row in min_normalize_time_idx.items()})

            min_index_time_idx = args.original_results['index_time'].idxmin(axis=0)
            raw_results['index_time'].loc[name] = pd.Series({col: args.original_results['index_time'].at[row, col] for col, row in min_index_time_idx.items()})

            for col, row in min_variate_idx.items():
                for metric in PHASE1_METRICS:
                    raw_results['simd_derived'].at[(name, metric), col] = args.original_results['simd_derived'].at[(row, metric), col]

        elif name == 'Worst':
            max_dominance_idx = args.original_results['dominance_checks'].idxmax(axis=0)
            raw_results['dominance_checks'].loc[name] = pd.Series({col: args.original_results['dominance_checks'].at[row, col] for col, row in max_dominance_idx.items()})

            max_variate_idx = args.original_results['variate_comparisons'].idxmax(axis=0)
            raw_results['variate_comparisons'].loc[name] = pd.Series({col: args.original_results['variate_comparisons'].at[row, col] for col, row in max_variate_idx.items()})

            max_simd_idx = args.original_results['simd_chunks'].idxmax(axis=0)
            raw_results['simd_chunks'].loc[name] = pd.Series({col: args.original_results['simd_chunks'].at[row, col] for col, row in max_simd_idx.items()})

            for col, row in max_variate_idx.items():
                for i in range(1, len(args.dataset.preference) + 1):
                    raw_results['vcheck_frequency'].at[(name, i), col] = args.original_results['vcheck_frequency'].at[(row, i), col]
            
            max_timing_idx = args.original_results['timing'].idxmax(axis=0)
            raw_results['timing'].loc[name] = pd.Series({col: args.original_results['timing'].at[row, col] for col, row in max_timing_idx.items()})

            max_parallel_idx = args.original_results['parallel_time'].idxmax(axis=0)
            raw_results['parallel_time'].loc[name] = pd.Series({col: args.original_results['parallel_time'].at[row, col] for col, row in max_parallel_idx.items()})

            max_compute_wall_idx = args.original_results['compute_wall'].idxmax(axis=0)
            raw_results['compute_wall'].loc[name] = pd.Series({col: args.original_results['compute_wall'].at[row, col] for col, row in max_compute_wall_idx.items()})

            max_python_overhead_idx = args.original_results['python_overhead'].idxmax(axis=0)
            raw_results['python_overhead'].loc[name] = pd.Series({col: args.original_results['python_overhead'].at[row, col] for col, row in max_python_overhead_idx.items()})

            max_sort_time_idx = args.original_results['sort_time'].idxmax(axis=0)
            raw_results['sort_time'].loc[name] = pd.Series({col: args.original_results['sort_time'].at[row, col] for col, row in max_sort_time_idx.items()})

            max_normalize_time_idx = args.original_results['normalize_time'].idxmax(axis=0)
            raw_results['normalize_time'].loc[name] = pd.Series({col: args.original_results['normalize_time'].at[row, col] for col, row in max_normalize_time_idx.items()})

            max_index_time_idx = args.original_results['index_time'].idxmax(axis=0)
            raw_results['index_time'].loc[name] = pd.Series({col: args.original_results['index_time'].at[row, col] for col, row in max_index_time_idx.items()})

            for col, row in max_variate_idx.items():
                for metric in PHASE1_METRICS:
                    raw_results['simd_derived'].at[(name, metric), col] = args.original_results['simd_derived'].at[(row, metric), col]

        elif name == 'Average':
            raw_results['dominance_checks'].loc[name] = args.original_results['dominance_checks'].mean(axis=0)
            raw_results['variate_comparisons'].loc[name] = args.original_results['variate_comparisons'].mean(axis=0)
            raw_results['simd_chunks'].loc[name] = args.original_results['simd_chunks'].mean(axis=0)
            raw_results['vcheck_frequency'] = pd.concat([
                raw_results['vcheck_frequency'],
                pd.concat(
                    {'Average': args.original_results['vcheck_frequency'].groupby(level=1).mean()},
                    names=args.original_results['vcheck_frequency'].index.names
                )
            ])
            raw_results['timing'].loc[name] = args.original_results['timing'].mean(axis=0)
            raw_results['parallel_time'].loc[name] = args.original_results['parallel_time'].mean(axis=0)
            raw_results['compute_wall'].loc[name] = args.original_results['compute_wall'].mean(axis=0)
            raw_results['python_overhead'].loc[name] = args.original_results['python_overhead'].mean(axis=0)
            raw_results['sort_time'].loc[name] = args.original_results['sort_time'].mean(axis=0)
            raw_results['normalize_time'].loc[name] = args.original_results['normalize_time'].mean(axis=0)
            raw_results['index_time'].loc[name] = args.original_results['index_time'].mean(axis=0)
            raw_results['simd_derived'] = pd.concat([
                raw_results['simd_derived'],
                pd.concat(
                    {'Average': args.original_results['simd_derived'].groupby(level=1).mean()},
                    names=args.original_results['simd_derived'].index.names
                )
            ])

        elif name == 'Median':
            raw_results['dominance_checks'].loc[name] = args.original_results['dominance_checks'].median(axis=0)
            raw_results['variate_comparisons'].loc[name] = args.original_results['variate_comparisons'].median(axis=0)
            raw_results['simd_chunks'].loc[name] = args.original_results['simd_chunks'].median(axis=0)
            raw_results['vcheck_frequency'] = pd.concat([
                raw_results['vcheck_frequency'],
                pd.concat(
                    {'Median': args.original_results['vcheck_frequency'].groupby(level=1).median()},
                    names=args.original_results['vcheck_frequency'].index.names
                )
            ])
            raw_results['timing'].loc[name] = args.original_results['timing'].median(axis=0)
            raw_results['parallel_time'].loc[name] = args.original_results['parallel_time'].median(axis=0)
            raw_results['compute_wall'].loc[name] = args.original_results['compute_wall'].median(axis=0)
            raw_results['python_overhead'].loc[name] = args.original_results['python_overhead'].median(axis=0)
            raw_results['sort_time'].loc[name] = args.original_results['sort_time'].median(axis=0)
            raw_results['normalize_time'].loc[name] = args.original_results['normalize_time'].median(axis=0)
            raw_results['index_time'].loc[name] = args.original_results['index_time'].median(axis=0)
            raw_results['simd_derived'] = pd.concat([
                raw_results['simd_derived'],
                pd.concat(
                    {'Median': args.original_results['simd_derived'].groupby(level=1).median()},
                    names=args.original_results['simd_derived'].index.names
                )
            ])

        else:
            order_key = list_repr(order)
            if order_key not in args.original_results['dominance_checks'].index:
                logging.warning("Skipping strategy %s because order %s is missing from original results.", name, order_key)
                continue

            raw_results['dominance_checks'].loc[name] = args.original_results['dominance_checks'].loc[order_key]
            raw_results['variate_comparisons'].loc[name] = args.original_results['variate_comparisons'].loc[order_key]
            raw_results['simd_chunks'].loc[name] = args.original_results['simd_chunks'].loc[order_key]
            raw_results['vcheck_frequency'] = pd.concat([
                raw_results['vcheck_frequency'],
                pd.concat(
                    {name: args.original_results['vcheck_frequency'].loc[order_key]},
                    names=args.original_results['vcheck_frequency'].index.names
                )
            ])
            raw_results['timing'].loc[name] = args.original_results['timing'].loc[order_key]
            raw_results['parallel_time'].loc[name] = args.original_results['parallel_time'].loc[order_key]
            raw_results['compute_wall'].loc[name] = args.original_results['compute_wall'].loc[order_key]
            raw_results['python_overhead'].loc[name] = args.original_results['python_overhead'].loc[order_key]
            raw_results['sort_time'].loc[name] = args.original_results['sort_time'].loc[order_key]
            raw_results['normalize_time'].loc[name] = args.original_results['normalize_time'].loc[order_key]
            raw_results['index_time'].loc[name] = args.original_results['index_time'].loc[order_key]
            raw_results['simd_derived'] = pd.concat([
                raw_results['simd_derived'],
                pd.concat(
                    {name: args.original_results['simd_derived'].loc[order_key]},
                    names=args.original_results['simd_derived'].index.names
                )
            ])

    return raw_results


def run_seed(args):
    set_state(args.state)

    args = Bunch(**args)
    args.excel_book = f"{args.excel_book}_{args.seed}"

    # compute correlation
    dataset: Dataset = args.dataset_class(dominance=args.dominance, size=n_samples, seed=args.seed)
    correl = dataset.data.corr()

    # check if original results are already cached
    original_results = load_original(args)

    if original_results is None:
        if len(dataset.preference) > HIGH_DIM_THRESHOLD:
            # Skip factorial permutation sweep — only run the three
            # correlation heuristic orderings (NCF/NACF/XCF). Dedup in
            # case two heuristics resolve to the same order.
            seen, orders = set(), []
            for _, order in get_strategy_orders(dataset):
                key = tuple(order)
                if key not in seen:
                    seen.add(key)
                    orders.append(list(order))
        else:
            orders = [list(x) for x in permutations(dataset.preference)]
        configs = [Bunch(**args, order=order, order_id=i) for i, order in enumerate(orders)]

        # variate_order is the leaf — no nested fan-out, so no submission
        # throttle is needed. The total number of in-flight leaves is already
        # bounded by ensemble_limit * seed_limit * len(orders).
        client = get_client()
        futures = client.map(variate_order, configs)
        results = client.gather(futures)

        original_results = agg_original(Bunch(**args, results=results))

        save_original(Bunch(**args, original_results=original_results))

    # reorganize based on strategies
    raw_results = strategies(Bunch(**args, original_results=original_results, dataset=dataset))

    # save the results
    save_seed(Bunch(**args, excel_sheet=f'Seed {args.seed}', raw_results=raw_results, dataset=dataset))

    raw_results['correl'] = correl

    return raw_results


def save_seed(args):
    with Excel(book_name=args.excel_book, folder_name='results/intermediate') as excel:
        excel.worksheet = args.excel_sheet

        # Stamp synth-dataset metadata (rank, init) from the dataset_class partial
        # onto the instance so display_info can report it.
        if hasattr(args, 'dataset_class'):
            for attr in ('rank', 'init'):
                value = getattr(args.dataset_class, attr, None)
                if value is not None and not hasattr(args.dataset, attr):
                    setattr(args.dataset, attr, value)

        display_info(args.dataset, excel=excel)

        plot_correlation(args.dataset, excel=excel)
        plot_correlation_preferences(args.dataset, excel=excel)

        plot_dataset_graph(args.dataset, excel=excel)

        display_results(args.dataset, args.raw_results, excel=excel)

        out_file = get_state().get_file('results/raw/dominance_checks', f'{args.excel_book}.csv')
        args.raw_results['dominance_checks'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/variate_comparisons', f'{args.excel_book}.csv')
        args.raw_results['variate_comparisons'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/simd_chunks', f'{args.excel_book}.csv')
        args.raw_results['simd_chunks'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/vcheck_frequency', f'{args.excel_book}.csv')
        args.raw_results['vcheck_frequency'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/timing', f'{args.excel_book}.csv')
        args.raw_results['timing'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/parallel_time', f'{args.excel_book}.csv')
        args.raw_results['parallel_time'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/compute_wall', f'{args.excel_book}.csv')
        args.raw_results['compute_wall'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/python_overhead', f'{args.excel_book}.csv')
        args.raw_results['python_overhead'].to_csv(out_file)

        out_file = get_state().get_file('results/raw/simd_derived', f'{args.excel_book}.csv')
        args.raw_results['simd_derived'].to_csv(out_file)


def variate_order(args) -> dict[str, pd.DataFrame]:
    # set the state
    set_state(args.state)

    # generate the dataset
    dataset: Dataset = args.dataset_class(dominance=args.dominance, size=n_samples, seed=args.seed)

    # set the variate
    dataset.preference = args.order

    # `lex_sort_key` selects the variate-ordering strategy stashed on the
    # dataset as the lex permutation key for the BNL/SFS/SaLSa/DnC lexsort
    # kernels. The dominance scan still walks columns in the strategy's own VO
    # (set above via args.order); None leaves lex_preference unset (matched-VO:
    # each strategy lex-sorts by its own VO column order).
    if any(resolve_lex_sort().values()):
        _lex_key_fns = {"XCF": algorithms.max_corr, "NCF": algorithms.min_corr}
        if lex_sort_key is None:
            pass  # matched-VO
        elif lex_sort_key in _lex_key_fns:
            dataset.lex_preference = _lex_key_fns[lex_sort_key](dataset)
        else:
            raise ValueError(
                f"invalid lex_sort_key={lex_sort_key!r}; expected 'XCF' | 'NCF' | None")

    # execute the experiment
    experiment: Experiment = Experiment(dataset)
    result = experiment.execute()

    return Bunch(**args, result=result)


class Experiment:

    def __init__(self, dataset):
        self.dataset = dataset

    def execute(self) -> dict[str, pd.DataFrame]:
        dominance_checks = []
        variate_comparisons = []
        simd_chunks = []
        vcheck_frequency = []
        timing = []
        parallel_time = []
        compute_wall = []
        python_overhead = []
        sort_time = []
        normalize_time = []
        index_time = []
        simd_derived_rows = []
        skylines = []

        # `_record` stamps one algorithm run's metrics into every result list
        # under `label`. `result` is the tuple returned by `_compute_skyline`.
        def _record(label, result):
            points, comparisons, sub_comparisons, simd_chunk_count, vcheck_freq, duration, p_duration, _, t_compute, t_python, t_sort, t_normalize, t_index, derived = result
            skylines.append(points)
            dominance_checks.append((label, comparisons))
            variate_comparisons.append((label, sub_comparisons))
            simd_chunks.append((label, simd_chunk_count if _simd_active else 0))
            vcheck_frequency.append((label, vcheck_freq))
            timing.append((label, duration))
            parallel_time.append((label, p_duration))
            compute_wall.append((label, t_compute))
            python_overhead.append((label, t_python))
            sort_time.append((label, t_sort))
            normalize_time.append((label, t_normalize))
            index_time.append((label, t_index))
            simd_derived_rows.append((label, *derived))

        if monotonic_experiment:
            # Monotonic-function analysis: run only SFS and SaLSa via the
            # monosort presort path (BNL, D&C, BBS skipped), once per configured
            # monotone function. The lexsort variants carry no monotone presort
            # and are bypassed regardless of `lex_sort`.
            if not sfs_monotonic and not salsa_monotonic:
                raise ValueError(
                    'monotonic_experiment is True but both sfs_monotonic and '
                    'salsa_monotonic are empty; nothing to run')
            for mono in sfs_monotonic:
                _record(f'SFS [{mono}]',
                        self._compute_skyline(SFSSkyline, monotonic_fn=mono))
            for mono in salsa_monotonic:
                _record(f'SaLSa [{mono}]',
                        self._compute_skyline(SaLSaSkyline, monotonic_fn=mono))
        else:
            # per-algorithm lexsort selection (bool broadcast or dict; see
            # skylines.resolve_lex_sort). Resolved here so a live mutation of
            # `skylines.lex_sort` (as repro scripts do) is picked up per run.
            lex = resolve_lex_sort()
            # which algorithms to run this experiment (see skylines.run_algorithms);
            # resolved here too so a live mutation is picked up per run. Returns
            # the selected labels in canonical lex_sort_algorithms order.
            selected = resolve_run_algorithms()
            # (plain class, lexsort class) per canonical algorithm label:
            #   BNL   — block nested loop      D&C — divide and conquer
            #   SFS   — sort filter skyline    BBS — branch and bound
            #   SaLSa — sort and limit skyline
            algo_classes = {
                'BNL':   (BNLSkyline,   BNLSkylineLexsort),
                'SFS':   (SFSSkyline,   SFSSkylineLexsort),
                'SaLSa': (SaLSaSkyline, SaLSaSkylineLexsort),
                'D&C':   (DnCSkyline,   DnCSkylineLexsort),
                'BBS':   (BBSSkyline,   BBSSkylineLexsort),
            }
            for label in selected:
                plain_cls, lexsort_cls = algo_classes[label]
                _record(label, self._compute_skyline(
                    lexsort_cls if lex[label] else plain_cls))

        # verify computed skyline
        verified = [df.equals(skylines[0]) for df in skylines]
        if not all(verified):
            raise AssertionError(f'The computed skylines are not identical: {verified}')

        # convert to dataframe
        dominance_checks = pd.DataFrame(dominance_checks, columns=['Model', 'Dominance Checks'])
        variate_comparisons = pd.DataFrame(variate_comparisons, columns=['Model', 'Variate Comparisons'])
        simd_chunks = pd.DataFrame(simd_chunks, columns=['Model', 'SIMD Chunks Compared'])
        timing = pd.DataFrame(timing, columns=['Model', 'Time (seconds)'])
        parallel_time = pd.DataFrame(parallel_time, columns=['Model', 'Time (seconds)'])
        compute_wall = pd.DataFrame(compute_wall, columns=['Model', 'Time (seconds)'])
        python_overhead = pd.DataFrame(python_overhead, columns=['Model', 'Time (seconds)'])
        sort_time = pd.DataFrame(sort_time, columns=['Model', 'Time (seconds)'])
        normalize_time = pd.DataFrame(normalize_time, columns=['Model', 'Time (seconds)'])
        index_time = pd.DataFrame(index_time, columns=['Model', 'Time (seconds)'])
        simd_derived = pd.DataFrame(simd_derived_rows, columns=['Model'] + PHASE1_METRICS)

        vcheck_frequency = pd.DataFrame.from_dict(dict(vcheck_frequency), orient='index').reset_index()
        vcheck_frequency = vcheck_frequency.rename(columns={'index': 'Model'})
        vcheck_frequency.columns = ['Model'] + list(range(1, vcheck_frequency.shape[1]))

        # create the results dictionary
        results = {'dominance_checks': dominance_checks, 'variate_comparisons': variate_comparisons, 'simd_chunks': simd_chunks, 'vcheck_frequency': vcheck_frequency, 'timing': timing, 'parallel_time': parallel_time, 'compute_wall': compute_wall, 'python_overhead': python_overhead, 'sort_time': sort_time, 'normalize_time': normalize_time, 'index_time': index_time, 'simd_derived': simd_derived}

        return results

    def _compute_skyline(self, skyline_class: Skyline.__class__, *args, **kwargs):
        skyline: Skyline = skyline_class(self.dataset, *args, **kwargs)
        points: pd.DataFrame = skyline.compute()
        derived = tuple(getattr(skyline, metric) for metric in PHASE1_METRICS)
        return points, skyline.comparisons, skyline.sub_comparisons, skyline.simd_chunks_compared, skyline.vcheck_freq, skyline.total_time, skyline.parallel_time, skyline.cluster_time, skyline.t_compute, skyline.t_python, skyline.sort_time, skyline.normalize_time, skyline.index_time, derived

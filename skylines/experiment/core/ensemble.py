import logging

from bunch_py3 import Bunch
import pandas as pd
from dask.distributed import get_client

from skylines.common.display.data import display_info
from skylines.common.display.result import display_ensemble_results
from skylines.common.excel import Excel, merge_workbooks
from skylines.common.state import get_state, set_state
from skylines import n_samples, n_runs, seed_limit
from skylines.dataset import Dataset
from skylines.experiment.core import windowed_map
from skylines.experiment.core.experiment import run_seed


def aggregate(args):
    set_state(args.state)

    average_result = None

    for result in args.results:
        if average_result is None:
            average_result = result
        else:
            average_result['dominance_checks'] += result['dominance_checks']
            average_result['variate_comparisons'] += result['variate_comparisons']
            average_result['simd_chunks'] += result['simd_chunks']
            average_result['vcheck_frequency'] += result['vcheck_frequency']
            average_result['timing'] += result['timing']
            average_result['parallel_time'] += result['parallel_time']
            average_result['compute_wall'] += result['compute_wall']
            average_result['python_overhead'] += result['python_overhead']
            average_result['sort_time'] += result['sort_time']
            average_result['normalize_time'] += result['normalize_time']
            average_result['index_time'] += result['index_time']
            average_result['simd_derived'] += result['simd_derived']
            average_result['correl'] += result['correl']

    average_result['dominance_checks'] = average_result['dominance_checks'].astype(float) / n_runs
    average_result['variate_comparisons'] = average_result['variate_comparisons'].astype(float) / n_runs
    average_result['simd_chunks'] = average_result['simd_chunks'].astype(float) / n_runs
    average_result['vcheck_frequency'] = average_result['vcheck_frequency'].astype(float) / n_runs
    average_result['timing'] = average_result['timing'].astype(float) / n_runs
    average_result['parallel_time'] = average_result['parallel_time'].astype(float) / n_runs
    average_result['compute_wall'] = average_result['compute_wall'].astype(float) / n_runs
    average_result['python_overhead'] = average_result['python_overhead'].astype(float) / n_runs
    average_result['sort_time'] = average_result['sort_time'].astype(float) / n_runs
    average_result['normalize_time'] = average_result['normalize_time'].astype(float) / n_runs
    average_result['index_time'] = average_result['index_time'].astype(float) / n_runs
    average_result['simd_derived'] = average_result['simd_derived'].astype(float) / n_runs
    average_result['correl'] = average_result['correl'].astype(float) / n_runs

    return Bunch(average_result)


def save(args):
    set_state(args.state)

    # save to work book
    dataset: Dataset = args.dataset_class(dominance=args.dominance, size=n_samples)

    # Stamp synth-dataset metadata (rank, init) from the dataset_class partial
    # onto the instance so display_info can report it.
    for attr in ('rank', 'init'):
        value = getattr(args.dataset_class, attr, None)
        if value is not None and not hasattr(dataset, attr):
            setattr(dataset, attr, value)

    # open excel
    with Excel(book_name=f'{args.excel_book}_summary', folder_name='results/intermediate') as excel:
        excel.worksheet = 'Summary'
        display_info(dataset, n_runs=n_runs, excel=excel)
        display_ensemble_results(args.result, excel=excel)

    # merge workbooks
    out_file = args.excel_book
    in_files = [f'{args.excel_book}_summary'] + [f'{args.excel_book}_{seed}' for seed in range(n_runs)]
    merge_workbooks(out_file=out_file, in_files=in_files)


def run_experiment(args):
    set_state(args.state)

    configs = [Bunch(**args, seed=i) for i in range(n_runs)]
    client = get_client()
    # Inner concurrency cap = `seed_limit`. Worker submitting these is the
    # throttle point — windowed_map keeps at most `seed_limit` `run_seed`
    # futures in-flight, so the scheduler cannot eager-distribute the full
    # n_runs batch and starve the variate_order leaves underneath them.
    results = list(windowed_map(client, run_seed, configs, window=seed_limit))

    result = aggregate(Bunch(**args, results=results))

    result = Bunch(**args, result=result)
    save(result)

    return result

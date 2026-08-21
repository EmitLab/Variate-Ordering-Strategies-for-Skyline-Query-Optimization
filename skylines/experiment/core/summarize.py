import numpy as np
import pandas as pd

from skylines import simd as _simd, n_samples

_simd_active = _simd != 'scalar'
from skylines.common.state import set_state
from skylines.common.utils import set_repr
from skylines.common.excel import Excel
from skylines.dataset import Dataset


def summarize(args):
    set_state(args.state)
    
    reverse_lookup = {value: key for key, values in args.run_experiment_sets.items() for value in values}

    agg_dominance_checks = dict()
    agg_variate_comparisons = dict()
    agg_simd_chunks = dict()
    agg_timing = dict()
    agg_parallel_time = dict()
    agg_compute_wall = dict()
    agg_python_overhead = dict()
    agg_sort_time = dict()
    agg_normalize_time = dict()
    agg_index_time = dict()
    agg_simd_derived = dict()

    agg_correl = dict()

    agg_vcheck_frequency = dict()
    
    dataset_name = dict()
    preferences = dict()
    pref_set_size = dict()
    cov_matrix_rank = dict()
    simd_settings = dict()

    name_set_map = dict()

    for run in args.runs:
        dataset: Dataset = run.dataset_class(dominance=run.dominance, size=n_samples)

        experiment_name = run.excel_book

        name_set_map[experiment_name] = reverse_lookup[run.experiment_no]

        dataset_name[experiment_name] = dataset.__class__.__name__
        preferences[experiment_name] = set_repr(frozenset(dataset.preference))
        pref_set_size[experiment_name] = len(dataset.preference)
        cov_matrix_rank[experiment_name] = getattr(run.dataset_class, 'rank', None)
        simd_settings[experiment_name] = _simd_active

        agg_dominance_checks[experiment_name] = run.result.dominance_checks
        agg_variate_comparisons[experiment_name] = run.result.variate_comparisons
        agg_simd_chunks[experiment_name] = run.result.simd_chunks
        agg_timing[experiment_name] = run.result.timing
        agg_parallel_time[experiment_name] = run.result.parallel_time
        agg_compute_wall[experiment_name] = run.result.get('compute_wall')
        agg_python_overhead[experiment_name] = run.result.get('python_overhead')
        agg_sort_time[experiment_name] = run.result.get('sort_time')
        agg_normalize_time[experiment_name] = run.result.get('normalize_time')
        agg_index_time[experiment_name] = run.result.get('index_time')
        agg_simd_derived[experiment_name] = run.result.get('simd_derived')

        agg_correl[experiment_name] = run.result.correl

        agg_vcheck_frequency[experiment_name] = run.result.vcheck_frequency

    # aggregate the results
    agg_dominance_checks = pd.concat(agg_dominance_checks)
    agg_variate_comparisons = pd.concat(agg_variate_comparisons)
    agg_simd_chunks = pd.concat(agg_simd_chunks)
    agg_timing = pd.concat(agg_timing)
    agg_parallel_time = pd.concat(agg_parallel_time)

    _compute_wall_present = all(v is not None for v in agg_compute_wall.values())
    _python_overhead_present = all(v is not None for v in agg_python_overhead.values())
    _sort_time_present = all(v is not None for v in agg_sort_time.values())
    _normalize_time_present = all(v is not None for v in agg_normalize_time.values())
    _index_time_present = all(v is not None for v in agg_index_time.values())
    _simd_derived_present = all(v is not None for v in agg_simd_derived.values())
    if _compute_wall_present:
        agg_compute_wall = pd.concat(agg_compute_wall)
    if _python_overhead_present:
        agg_python_overhead = pd.concat(agg_python_overhead)
    if _sort_time_present:
        agg_sort_time = pd.concat(agg_sort_time)
    if _normalize_time_present:
        agg_normalize_time = pd.concat(agg_normalize_time)
    if _index_time_present:
        agg_index_time = pd.concat(agg_index_time)
    if _simd_derived_present:
        agg_simd_derived = pd.concat(agg_simd_derived)

    # generate result analysis
    causal_cols = [col for col in agg_dominance_checks.columns if 'Causal' in col]
    agg_dominance_checks_perc = pd.DataFrame()
    for col in causal_cols:
        base_col = col.split(' Causal')[0]
        if base_col in agg_dominance_checks:
            agg_dominance_checks_perc[col] = ((agg_dominance_checks[col] - agg_dominance_checks[base_col]) / agg_dominance_checks[base_col])
    
    causal_cols = [col for col in agg_variate_comparisons.columns if 'Causal' in col]
    agg_variate_comparisons_perc = pd.DataFrame()
    for col in causal_cols:
        base_col = col.split(' Causal')[0]
        if base_col in agg_variate_comparisons:
            agg_variate_comparisons_perc[col] = ((agg_variate_comparisons[col] - agg_variate_comparisons[base_col]) / agg_variate_comparisons[base_col])

    causal_cols = [col for col in agg_timing.columns if 'Causal' in col]
    agg_timing_perc = pd.DataFrame()
    for col in causal_cols:
        base_col = col.split(' Causal')[0]
        if base_col in agg_timing:
            agg_timing_perc[col] = ((agg_timing[col] - agg_timing[base_col]) / agg_timing[base_col])

    causal_cols = [col for col in agg_parallel_time.columns if 'Causal' in col]
    agg_parallel_time_perc = pd.DataFrame()
    for col in causal_cols:
        base_col = col.split(' Causal')[0]
        if base_col in agg_parallel_time:
            agg_parallel_time_perc[col] = ((agg_parallel_time[col] - agg_parallel_time[base_col]) / agg_parallel_time[base_col])

    # write to excel
    with Excel('Summary') as excel:
        excel.worksheet = 'Summary'

        summary_headers = ['Experiment Set', 'Experiment', 'Dataset', 'Preferences', 'Pref Set Size', 'Cov Matrix Rank', 'Strategy', 'Corr Mean', 'Corr Std', 'SIMD']
        for col, header in enumerate(summary_headers):
            excel.write_text(header, bold=True, startrow=1, startcol=col)

        for row, name in enumerate(agg_dominance_checks.index):
            excel.write_text(name_set_map[name[0]], startrow=row + 2, startcol=0)
            excel.write_text(name[0], startrow=row + 2, startcol=1)
            excel.write_text(dataset_name[name[0]], startrow=row + 2, startcol=2)
            excel.write_text(preferences[name[0]], startrow=row + 2, startcol=3)
            excel.write_text(pref_set_size[name[0]], startrow=row + 2, startcol=4)
            rank = cov_matrix_rank[name[0]]
            excel.write_text('' if rank is None else rank, startrow=row + 2, startcol=5)
            excel.write_text(name[1], startrow=row + 2, startcol=6)

            correl = agg_correl[name[0]]
            mask = ~np.eye(len(correl), dtype=bool)
            row_means = correl.where(mask).mean(axis=1)
            excel.write_text(row_means.mean(), startrow=row + 2, startcol=7)
            excel.write_text(row_means.std(), startrow=row + 2, startcol=8)
            excel.write_text('Yes' if simd_settings[name[0]] else 'No', startrow=row + 2, startcol=9)

        benchmark_col = len(summary_headers) + 1
        excel.write_dataframe(agg_dominance_checks, 0, benchmark_col, index=False, title='Average Benchmark (dominance checks)', color=True, invert_color=True)
        excel.write_dataframe(agg_variate_comparisons, 0, benchmark_col + agg_dominance_checks.shape[1] + 1, index=False, title='Average Benchmark (variate comparisons)', color=True, invert_color=True)
        excel.write_dataframe(agg_simd_chunks, 0, benchmark_col + agg_dominance_checks.shape[1] + agg_variate_comparisons.shape[1] + 2, index=False, title='Average Benchmark (SIMD chunks compared)', color=True, invert_color=True)
        excel.write_dataframe(agg_timing, 0, benchmark_col + agg_dominance_checks.shape[1] + agg_variate_comparisons.shape[1] + agg_simd_chunks.shape[1] + 3, index=False, title='Average Benchmark (timing)', color=True, invert_color=True)
        excel.write_dataframe(agg_parallel_time, 0, benchmark_col + agg_dominance_checks.shape[1] + agg_variate_comparisons.shape[1] + agg_simd_chunks.shape[1] + agg_timing.shape[1] + 4, index=False, title='Average Benchmark (parallel timing)', color=True, invert_color=True)

        _col_after_parallel = benchmark_col + agg_dominance_checks.shape[1] + agg_variate_comparisons.shape[1] + agg_simd_chunks.shape[1] + agg_timing.shape[1] + agg_parallel_time.shape[1] + 5
        if _compute_wall_present:
            excel.write_dataframe(agg_compute_wall, 0, _col_after_parallel, index=False, title='Average Benchmark (compute wall)', color=True, invert_color=True)
            _col_after_compute = _col_after_parallel + agg_compute_wall.shape[1] + 1
            if _python_overhead_present:
                excel.write_dataframe(agg_python_overhead, 0, _col_after_compute, index=False, title='Average Benchmark (python overhead)', color=True, invert_color=True)
                _col_after_compute = _col_after_compute + agg_python_overhead.shape[1] + 1
            if _sort_time_present:
                excel.write_dataframe(agg_sort_time, 0, _col_after_compute, index=False, title='Average Benchmark (sort timing)', color=True, invert_color=True)
                _col_after_compute = _col_after_compute + agg_sort_time.shape[1] + 1
            if _normalize_time_present:
                excel.write_dataframe(agg_normalize_time, 0, _col_after_compute, index=False, title='Average Benchmark (normalize time)', color=True, invert_color=True)
                _col_after_compute = _col_after_compute + agg_normalize_time.shape[1] + 1
            if _index_time_present:
                excel.write_dataframe(agg_index_time, 0, _col_after_compute, index=False, title='Average Benchmark (indexing timing)', color=True, invert_color=True)

        if _simd_derived_present:
            excel.worksheet = 'SIMD Derived Metrics'
            excel.write_dataframe(agg_simd_derived.reset_index(), 0, 0, index=False, title='SIMD Derived Metrics', color=True, invert_color=True)

        excel.worksheet = 'Correlations'

        row = 0
        for name, correl in agg_correl.items():
            excel.write_text(name, startrow=row + 1, startcol=0)
            excel.write_dataframe(correl, row, 2, index=True, title='Original Correlation', color=True)
            row += correl.shape[1] + 5

        excel.worksheet = 'vcheck Frequency'

        # col = 0
        # for name, freq in agg_vcheck_frequency.items():
        #     excel.write_text(name, startrow=0, startcol=col + 1)
        #     excel.write_dataframe(freq, 2, col + 1, index=True, title='vcheck Frequency', color=True, invert_color=True)
        #     col += freq.shape[1] + 3

        agg_vcheck_frequency = pd.concat(agg_vcheck_frequency, names=['Experiment']).reset_index()
        excel.write_dataframe(agg_vcheck_frequency, 0, 0, index=False, title='vcheck Frequency', color=True, invert_color=True)

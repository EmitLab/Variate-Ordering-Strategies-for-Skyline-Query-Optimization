from itertools import permutations
import pandas as pd
from bunch_py3 import Bunch

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
from skylines import lex_sort

import warnings
warnings.filterwarnings('ignore')


class Experiment:

    def __init__(self, dataset: Dataset, excel_book: str, excel_sheet: str):
        self.dataset = dataset
        self.excel_book = excel_book
        self.excel_sheet = excel_sheet
        self.memoise = Bunch()

    def execute(self, state: State, silent: bool = False, progress = None) -> dict[str, pd.DataFrame]:
        # set the state
        set_state(state)

        # compute data
        correl = self.dataset.data.corr()

        # check if original results are already cached
        raw_results = self._load_original()

        if raw_results is None:
            # run sub-experiment for each confounder
            raw_results: dict = {'dominance_checks': None, 'variate_comparisons': None, 'timing': None, 'parallel_time': None}
            
            # get all permutations
            orders = [list(x) for x in permutations(self.dataset.preference)]

            # store the original preference order
            original_pref = self.dataset.preference

            # execute experiments
            for subindex, order in enumerate(orders):
                # execute the experiment
                self.dataset.preference = order
                cluster_result: dict[str, pd.DataFrame] = self._execute()

                if progress is not None:
                    progress(subindex / len(orders))

                # close all plots to reduce resource consumption
                plt.close('all')

                # add cluster results to all results
                if raw_results['dominance_checks'] is None:
                    raw_results['dominance_checks'] = pd.DataFrame(columns=cluster_result['dominance_checks']['Model'].values)
                    raw_results['dominance_checks'].index.name = 'Cluster'
                raw_results['dominance_checks'].loc[list_repr(order)] = cluster_result['dominance_checks']['Dominance Checks'].values

                if raw_results['variate_comparisons'] is None:
                    raw_results['variate_comparisons'] = pd.DataFrame(columns=cluster_result['variate_comparisons']['Model'].values)
                    raw_results['variate_comparisons'].index.name = 'Cluster'
                raw_results['variate_comparisons'].loc[list_repr(order)] = cluster_result['variate_comparisons']['Variate Comparisons'].values
                
                if raw_results['timing'] is None:
                    raw_results['timing'] = pd.DataFrame(columns=cluster_result['timing']['Model'].values)
                    raw_results['timing'].index.name = 'Cluster'
                raw_results['timing'].loc[list_repr(order)] = cluster_result['timing']['Time (seconds)'].values

                if raw_results['parallel_time'] is None:
                    raw_results['parallel_time'] = pd.DataFrame(columns=cluster_result['parallel_time']['Model'].values)
                    raw_results['parallel_time'].index.name = 'Cluster'
                raw_results['parallel_time'].loc[list_repr(order)] = cluster_result['parallel_time']['Time (seconds)'].values

            # store the expanded raw/original results
            self._save_original(raw_results)

            # restore the original preference order
            self.dataset.preference = original_pref

        # rearrange results
        results: dict = {'dominance_checks': raw_results['dominance_checks'].iloc[0:0],
                         'variate_comparisons': raw_results['variate_comparisons'].iloc[0:0],
                         'timing': raw_results['timing'].iloc[0:0],
                         'parallel_time': raw_results['parallel_time'].iloc[0:0],
                         'correl': correl}

        heuristics = [('Best', None),
                      ('Worst', None),
                      ('Average', None),
                      ('Median', None),
                      ('NC_best', algorithms.min_corr(self.dataset)),
                      ('ANC_best', algorithms.min_abs_corr(self.dataset)),
                      ('XC_best', algorithms.max_corr(self.dataset)),
                      ('NC_pairwise_best', algorithms.min_corr_pairwise(self.dataset)),
                      ('INCF_best', algorithms.incremental_min_corr_first(self.dataset)),
                      ('IXCL_best', algorithms.incremental_max_corr_last(self.dataset))]
        
        for name, order in heuristics:
            if name == 'Best':
                min_idx = raw_results['dominance_checks'].idxmin(axis=0)
                results['dominance_checks'].loc[name] = pd.Series({col: raw_results['dominance_checks'].at[row, col] for col, row in min_idx.items()})

                min_idx = raw_results['variate_comparisons'].idxmin(axis=0)
                results['variate_comparisons'].loc[name] = pd.Series({col: raw_results['variate_comparisons'].at[row, col] for col, row in min_idx.items()})

                min_idx = raw_results['timing'].idxmin(axis=0)
                results['timing'].loc[name] = pd.Series({col: raw_results['timing'].at[row, col] for col, row in min_idx.items()})

                min_idx = raw_results['parallel_time'].idxmin(axis=0)
                results['parallel_time'].loc[name] = pd.Series({col: raw_results['parallel_time'].at[row, col] for col, row in min_idx.items()})
                
            elif name == 'Worst':
                max_idx = raw_results['dominance_checks'].idxmax(axis=0)
                results['dominance_checks'].loc[name] = pd.Series({col: raw_results['dominance_checks'].at[row, col] for col, row in max_idx.items()})

                max_idx = raw_results['variate_comparisons'].idxmax(axis=0)
                results['variate_comparisons'].loc[name] = pd.Series({col: raw_results['variate_comparisons'].at[row, col] for col, row in max_idx.items()})
                
                max_idx = raw_results['timing'].idxmax(axis=0)
                results['timing'].loc[name] = pd.Series({col: raw_results['timing'].at[row, col] for col, row in max_idx.items()})

                max_idx = raw_results['parallel_time'].idxmax(axis=0)
                results['parallel_time'].loc[name] = pd.Series({col: raw_results['parallel_time'].at[row, col] for col, row in max_idx.items()})
            
            elif name == 'Average':
                results['dominance_checks'].loc[name] = raw_results['dominance_checks'].mean(axis=0)
                results['variate_comparisons'].loc[name] = raw_results['variate_comparisons'].mean(axis=0)
                results['timing'].loc[name] = raw_results['timing'].mean(axis=0)
                results['parallel_time'].loc[name] = raw_results['parallel_time'].mean(axis=0)
            
            elif name == 'Median':
                results['dominance_checks'].loc[name] = raw_results['dominance_checks'].median(axis=0)
                results['variate_comparisons'].loc[name] = raw_results['variate_comparisons'].median(axis=0)
                results['timing'].loc[name] = raw_results['timing'].median(axis=0)
                results['parallel_time'].loc[name] = raw_results['parallel_time'].median(axis=0)

            else:
                results['dominance_checks'].loc[name] = raw_results['dominance_checks'].loc[list_repr(order)]
                results['variate_comparisons'].loc[name] = raw_results['variate_comparisons'].loc[list_repr(order)]
                results['timing'].loc[name] = raw_results['timing'].loc[list_repr(order)]
                results['parallel_time'].loc[name] = raw_results['parallel_time'].loc[list_repr(order)]

        # save the results
        if not silent:
            self._save(results)

        return results
    
    def _load_original(self):
        try:
            raw_results = dict()

            in_file = get_state().get_file('results/original/dominance_checks', f'{self.excel_book}.csv')
            raw_results['dominance_checks'] = pd.read_csv(in_file, index_col=0)

            in_file = get_state().get_file('results/original/variate_comparisons', f'{self.excel_book}.csv')
            raw_results['variate_comparisons'] = pd.read_csv(in_file, index_col=0)

            in_file = get_state().get_file('results/original/timing', f'{self.excel_book}.csv')
            raw_results['timing'] = pd.read_csv(in_file, index_col=0)

            in_file = get_state().get_file('results/original/parallel_time', f'{self.excel_book}.csv')
            raw_results['parallel_time'] = pd.read_csv(in_file, index_col=0)

            return raw_results
        
        except Exception:
            return None
    
    def _save_original(self, raw_results: dict[str: pd.DataFrame]):
        out_file = get_state().get_file('results/original/dominance_checks', f'{self.excel_book}.csv')
        raw_results['dominance_checks'].to_csv(out_file)

        out_file = get_state().get_file('results/original/variate_comparisons', f'{self.excel_book}.csv')
        raw_results['variate_comparisons'].to_csv(out_file)

        out_file = get_state().get_file('results/original/timing', f'{self.excel_book}.csv')
        raw_results['timing'].to_csv(out_file)

        out_file = get_state().get_file('results/original/parallel_time', f'{self.excel_book}.csv')
        raw_results['parallel_time'].to_csv(out_file)

    def _save(self, results: dict[str, pd.DataFrame]):
        with Excel(book_name=self.excel_book, folder_name='results/intermediate') as excel:
            excel.worksheet = self.excel_sheet

            display_info(self.dataset, excel=excel)

            plot_correlation(self.dataset, excel=excel)
            plot_correlation_preferences(self.dataset, excel=excel)

            plot_dataset_graph(self.dataset, excel=excel)

            display_results(self.dataset, results, excel=excel)

            out_file = get_state().get_file('results/raw/dominance_checks', f'{self.excel_book}.csv')
            results['dominance_checks'].to_csv(out_file)

            out_file = get_state().get_file('results/raw/variate_comparisons', f'{self.excel_book}.csv')
            results['variate_comparisons'].to_csv(out_file)

            out_file = get_state().get_file('results/raw/timing', f'{self.excel_book}.csv')
            results['timing'].to_csv(out_file)

            out_file = get_state().get_file('results/raw/parallel_time', f'{self.excel_book}.csv')
            results['parallel_time'].to_csv(out_file)

    def _execute(self) -> dict[str, pd.DataFrame]:
        dominance_checks = []
        variate_comparisons = []
        timing = []
        parallel_time = []
        skylines = []

        # find the skyline using block nested loop approach
        points, comparisons, sub_comparisons, duration, p_duration, _ = self._compute_skyline(BNLSkylineLexsort if lex_sort else BNLSkyline, 'BNL', memoise=False)
        skylines.append(points)
        dominance_checks.append(('BNL', comparisons))
        variate_comparisons.append(('BNL', sub_comparisons))
        timing.append(('BNL', duration))
        parallel_time.append(('BNL', p_duration))

        # find the skyline using sort filter approach
        points, comparisons, sub_comparisons, duration, p_duration, _ = self._compute_skyline(SFSSkylineLexsort if lex_sort else SFSSkyline, 'SFS', memoise=False)
        skylines.append(points)
        dominance_checks.append(('SFS', comparisons))
        variate_comparisons.append(('SFS', sub_comparisons))
        timing.append(('SFS', duration))
        parallel_time.append(('SFS', p_duration))

        # find the skyline using SaLSa approach
        points, comparisons, sub_comparisons, duration, p_duration, _ = self._compute_skyline(SaLSaSkylineLexsort if lex_sort else SaLSaSkyline, 'SaLSa', memoise=False)
        skylines.append(points)
        dominance_checks.append(('SaLSa', comparisons))
        variate_comparisons.append(('SaLSa', sub_comparisons))
        timing.append(('SaLSa', duration))
        parallel_time.append(('SaLSa', p_duration))

        # find the skyline using divide and conquer approach
        points, comparisons, sub_comparisons, duration, p_duration, _ = self._compute_skyline(DnCSkylineLexsort if lex_sort else DnCSkyline, 'D&C', memoise=False)
        skylines.append(points)
        dominance_checks.append(('D&C', comparisons))
        variate_comparisons.append(('D&C', sub_comparisons))
        timing.append(('D&C', duration))
        parallel_time.append(('D&C', p_duration))

        # # find the skyline using branch and bound approach
        # points, comparisons, sub_comparisons, duration, p_duration, _ = self._compute_skyline(BBSSkylineLexsort if lex_sort else BBSSkyline, 'BBS', memoise=False)
        # skylines.append(points)
        # dominance_checks.append(('BBS', comparisons))
        # variate_comparisons.append(('BBS', sub_comparisons))
        # timing.append(('BBS', duration))
        # parallel_time.append(('BBS', p_duration))

        # verify computed skyline
        verified = [df.equals(skylines[0]) for df in skylines]
        if not all(verified):
            raise AssertionError(f'The computed skylines are not identical: {verified}')

        # convert to dataframe
        dominance_checks = pd.DataFrame(dominance_checks, columns=['Model', 'Dominance Checks'])
        variate_comparisons = pd.DataFrame(variate_comparisons, columns=['Model', 'Variate Comparisons'])
        timing = pd.DataFrame(timing, columns=['Model', 'Time (seconds)'])
        parallel_time = pd.DataFrame(parallel_time, columns=['Model', 'Time (seconds)'])

        # create the results dictionary
        results = {'dominance_checks': dominance_checks, 'variate_comparisons': variate_comparisons, 'timing': timing, 'parallel_time': parallel_time}

        return results

    def _compute_skyline(self, skyline_class: Skyline.__class__, title: str, memoise: bool = False, *args, **kwargs):
        def _compute():
            skyline: Skyline = skyline_class(self.dataset, *args, **kwargs)
            points: pd.DataFrame = skyline.compute()
            return points, skyline.comparisons, skyline.sub_comparisons, skyline.total_time, skyline.parallel_time, skyline.cluster_time

        if memoise:
            if title not in self.memoise:
                memoir = Bunch()
                memoir.points, memoir.comparisons, memoir.sub_comparisons, memoir.total_time, memoir.parallel_time, memoir.cluster_time = _compute()
                self.memoise[title] = memoir

            memoir = self.memoise[title]
            return memoir.points, memoir.comparisons, memoir.sub_comparisons, memoir.total_time, memoir.parallel_time, memoir.cluster_time

        return _compute()

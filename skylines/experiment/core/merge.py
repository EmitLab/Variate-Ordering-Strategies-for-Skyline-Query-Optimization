import pandas as pd
from tqdm import tqdm
from skylines.common.constants.constants import get_experiments
from skylines.common.state import get_state, set_state
from skylines import n_samples, n_runs, experiment_type
from skylines.common.state.state import State
from skylines.common.utils.utils import format_num
from skylines.dataset.dataset import Dataset


def _load_original(excel_book, suffix):
    try:
        raw_results = dict()

        in_file = get_state().get_file(f'results/original_{suffix}/dominance_checks', f'{excel_book}.csv')
        raw_results['dominance_checks'] = pd.read_csv(in_file, index_col=0)

        in_file = get_state().get_file(f'results/original_{suffix}/variate_comparisons', f'{excel_book}.csv')
        raw_results['variate_comparisons'] = pd.read_csv(in_file, index_col=0)

        in_file = get_state().get_file(f'results/original_{suffix}/timing', f'{excel_book}.csv')
        raw_results['timing'] = pd.read_csv(in_file, index_col=0)

        in_file = get_state().get_file(f'results/original_{suffix}/parallel_time', f'{excel_book}.csv')
        raw_results['parallel_time'] = pd.read_csv(in_file, index_col=0)

        return raw_results
    
    except Exception:
        return None


def _save_original(excel_book, raw_results: dict[str: pd.DataFrame]):
    out_file = get_state().get_file('results/original/dominance_checks', f'{excel_book}.csv')
    raw_results['dominance_checks'].to_csv(out_file)

    out_file = get_state().get_file('results/original/variate_comparisons', f'{excel_book}.csv')
    raw_results['variate_comparisons'].to_csv(out_file)

    out_file = get_state().get_file('results/original/timing', f'{excel_book}.csv')
    raw_results['timing'].to_csv(out_file)

    out_file = get_state().get_file('results/original/parallel_time', f'{excel_book}.csv')
    raw_results['parallel_time'].to_csv(out_file)


def replace(seed_book_name, old_results, new_results):
    old_results['dominance_checks']['D&C'] = new_results['dominance_checks']['D&C']
    old_results['variate_comparisons']['D&C'] = new_results['variate_comparisons']['D&C']
    old_results['timing']['D&C'] = new_results['timing']['D&C']
    old_results['parallel_time']['D&C'] = new_results['parallel_time']['D&C']

    _save_original(seed_book_name, old_results)


def merge(run_experiments):
    # n_experiment = sum([len(get_experiments()[experiment_no]) for experiment_no in run_experiments])

    samples = format_num(n_samples)

    for experiment_no in tqdm(run_experiments):
        # output details
        datasets = get_experiments()[experiment_no]

        for dataset_class, dominance in datasets:
            # create an instance of the dataset
            dataset: Dataset = dataset_class(dominance=dominance, size=n_samples)

            # create excel book name
            book_name = f'Experiment_{experiment_no}_{dataset.dominance_key}_{samples}'

            for seed in range(n_runs):
                seed_book_name = f'{book_name}_{seed}'

                old_results = _load_original(seed_book_name, 'old')
                new_results = _load_original(seed_book_name, 'new')

                replace(seed_book_name, old_results, new_results)


def main():
    # Initialize state
    state: State = State(n_samples)
    set_state(state)

    # Select experiments to run
    if experiment_type == 'Synthetic - 1':
        # {2, 3, 4, 5}, 200K, Mixed, Wide
        run_experiments = [i for i in range(166, 186)] + [i for i in range(201, 206)]
        
    elif experiment_type == 'Synthetic - 2':
        # 4, {50K, 100K, 200K, 400K}, Mixed, Wide
        run_experiments = [i for i in range(181, 186)]
        
    elif experiment_type == 'Synthetic - 3':
        # 4, 200K, {Positive, Mixed, Negative}, Wide
        run_experiments = [i for i in range(191, 196)] \
            + [i for i in range(186, 191)] \
            + [i for i in range(181, 186)]
    
    elif experiment_type == 'Synthetic - 4':
        # 4, 200K, Mixed, {Narrow, Wide}
        run_experiments = [i for i in range(196, 201)] + [i for i in range(181, 186)]
    
    elif experiment_type == 'Real - 1':
        # {2, 3, 4, 5}, 200K, Mixed, Wide
        run_experiments = [i for i in range(206, 246)]
        
    elif experiment_type == 'Real - 2':
        # 4, {50K, 100K, 200K, 400K}, Mixed, Wide
        run_experiments = [i for i in range(230, 238)]
    
    elif experiment_type == 'Synthetic - New Plan':
        # run_experiments = [i for i in range(246, 258)]
        run_experiments = [i for i in range(249, 253)]
        
    else:
        print('Invalid experiment type')
        exit(1)

    merge(run_experiments)


if __name__ == '__main__':
    main()

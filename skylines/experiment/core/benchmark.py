import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from argparse import ArgumentParser
from itertools import chain

import dask
dask.config.set({
    "distributed.scheduler.worker-ttl": "1h",
    "distributed.comm.timeouts.connect": "300s",
    "distributed.scheduler.dashboard.bokeh-application.session_token_expiration": 86400,
    })

from bunch_py3 import Bunch
from dask.distributed import Client
from tqdm.auto import tqdm

import skylines
from skylines.common.constants import get_experiments
from skylines.common.state import State, set_state
from skylines import n_samples, experiment_type, n_workers, ensemble_limit
from skylines.common.utils import dataset_label, format_num
from skylines.dataset import Dataset
from skylines.experiment.core import windowed_map
from skylines.experiment.core.ensemble import run_experiment
from skylines.experiment.core.summarize import summarize


def experiment_flow(args):
    set_state(args.state)

    samples = format_num(n_samples)

    configs = []

    run_experiments = list(chain.from_iterable(args.run_experiment_sets.values()))

    for experiment_no in run_experiments:
        # output details
        datasets = get_experiments()[experiment_no]

        for dataset_class, dominance in datasets:
            # create an instance of the dataset
            dataset: Dataset = dataset_class(dominance=dominance, size=n_samples)

            # create excel book name
            extras = dataset_label(dataset_class)
            if len(extras) == 0:
                book_name = f'Experiment_{experiment_no}_{dataset.dominance_key}_{samples}'
            else:
                book_name = f'Experiment_{experiment_no}_{"_".join(extras)}_{dataset.dominance_key}_{samples}'

            # create the experiment
            config = Bunch()
            config.state = args.state
            config.experiment_no = experiment_no
            config.dataset_class = dataset_class
            config.dominance = dominance
            config.excel_book = book_name
            config.experiment_name = book_name

            configs.append(config)

    with Client(n_workers=n_workers, threads_per_worker=1, processes=True) as client:
        print(f"Dask dashboard: {client.dashboard_link}", flush=True)

        # Outer concurrency cap = `ensemble_limit`. Driver-side window throttles
        # submission so the scheduler never queues more than `ensemble_limit`
        # in-flight `run_experiment` futures, leaving worker slots free for the
        # inner `run_seed` / `variate_order` tasks each outer fans out to.
        runs = list(
            tqdm(
                windowed_map(client, run_experiment, configs, window=ensemble_limit),
                total=len(configs),
                desc="experiments",
            )
        )

    summarize(Bunch(**args, runs=runs))


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--no-refresh", action="store_true", help="Disable timestamp refresh")
    args = parser.parse_args()

    # Initialize state
    state: State = State(n_samples)
    set_state(state)

    # Refresh state
    if not args.no_refresh:
        state.refresh()

    # Snapshot config (skylines/__init__.py) into the timestamp folder
    state.snapshot_config(skylines.__file__)

    # Select experiments to run
    if experiment_type == 'Synthetic - 1':
        # {2, 3, 4, 5}, 200K, Mixed, Wide
        run_experiment_sets = {
            31: [i for i in range(166, 176)],
            32: [i for i in range(176, 181)],
            33: [i for i in range(181, 186)],
            40: [i for i in range(201, 206)]
        }

    elif experiment_type == 'Synthetic - 2':
        # 4, {50K, 100K, 200K, 400K}, Mixed, Wide
        run_experiment_sets = {
            35: [i for i in range(181, 186)],
        }
    
    elif experiment_type == 'Synthetic - 3':
        # 4, 200K, {Positive, Mixed, Negative}, Wide
        run_experiment_sets = {
            36: [i for i in range(186, 191)],  # positive
            41: [i for i in range(181, 186)],  # mixed
            37: [i for i in range(191, 196)]   # negative
        }
    
    elif experiment_type == 'Synthetic - 4':
        # 4, 200K, Mixed, {Narrow, Wide}
        run_experiment_sets = {
            38: [i for i in range(196, 201)],
            39: [i for i in range(181, 186)]
        }

    elif experiment_type == 'Real - 1':
        # {2, 3, 4, 5}, 200K, Mixed, Wide
        run_experiment_sets = {
            42: [i for i in range(206, 222)],
            43: [i for i in range(222, 230)],
            44: [i for i in range(230, 238)],
            45: [i for i in range(238, 246)]
        }
        
    elif experiment_type == 'Real - 2':
        # 4, {50K, 100K, 200K, 400K}, Mixed, Wide
        run_experiment_sets = {
            46: [i for i in range(230, 238)],
        }
    
    elif experiment_type == 'Synthetic - New Plan':
        run_experiment_sets = {
            47: [i for i in range(246, 249)],
            48: [i for i in range(249, 253)],
            49: [i for i in range(253, 258)]
        }
    
    elif experiment_type == 'Synthetic - New Plan - 4 Vars':
        run_experiment_sets = {
            48: [i for i in range(249, 253)],
        }

    elif experiment_type == 'Synthetic - 24 variates':
        run_experiment_sets = {
            50: [i for i in range(258, 260)]
        }

    elif experiment_type == 'Smoke - 3 variates':
        # Issue #1 Phase 0 smoke — Table 6 row: |P|=3, rank=1.
        # Toggle lex_sort in skylines/__init__.py between runs.
        run_experiment_sets = {
            46: [246],
        }

    elif experiment_type == 'Smoke - 5 variates':
        # Issue #1 Phase 0 smoke — Table 7 row: |P|=5, rank=1.
        # Toggle lex_sort / simd / dtype in skylines/__init__.py between runs.
        run_experiment_sets = {
            53: [253],
        }

    elif experiment_type == 'CelebA':
        # 40 facial attribute objectives, all MAX, 50 K samples
        run_experiment_sets = {
            300: [300]
        }

    elif experiment_type == 'Real - New':
        # New VOS real datasets — all standard variants (2–5 preference variables)
        # Datasets: NBA, Weather, CASP, BreastCancer, Parkinsons,
        #           Pol, Elevators, SkillCraft, KEGG  (QWS excluded: manual download needed)
        run_experiment_sets = {
            51: [301, 306, 316, 321, 329, 334, 341, 347, 352],  # 2var pos
            52: [302, 307, 317, 322, 330, 335, 342, 348, 353],  # 2var neg
            53: [303, 308, 318, 323, 331, 336, 343, 349, 354],  # 3var mixed
            54: [304, 309, 319, 324, 332, 337, 344, 350, 355],  # 4var mixed
            55: [305, 310, 320, 325, 333, 338, 345, 351, 356],  # 5var mixed
        }

    elif experiment_type == 'Real - New - HighDim':
        # High-dimensional variants of new datasets
        # BreastCancer 10/20/30var, Pol 10/26var, Elevators 18var, KEGG 10/27var
        run_experiment_sets = {
            56: [326, 327, 328, 339, 340, 346, 357, 358],
        }

    elif experiment_type == 'Real - New - NBA':
        run_experiment_sets = {
            57: list(range(301, 306)),
        }

    elif experiment_type == 'Real - New - Weather':
        run_experiment_sets = {
            58: list(range(306, 311)),
        }

    elif experiment_type == 'Real - New - CASP':
        run_experiment_sets = {
            59: list(range(316, 321)),
        }

    elif experiment_type == 'Real - New - BreastCancer':
        run_experiment_sets = {
            60: list(range(321, 329)),  # standard + high-dim (2var through 30var)
        }

    elif experiment_type == 'Real - New - Parkinsons':
        run_experiment_sets = {
            61: list(range(329, 334)),
        }

    elif experiment_type == 'Real - New - Pol':
        run_experiment_sets = {
            62: list(range(334, 341)),  # standard + high-dim
        }

    elif experiment_type == 'Real - New - Elevators':
        run_experiment_sets = {
            63: list(range(341, 347)),  # standard + high-dim
        }

    elif experiment_type == 'Real - New - SkillCraft':
        run_experiment_sets = {
            64: list(range(347, 352)),
        }

    elif experiment_type == 'Real - New - KEGG':
        run_experiment_sets = {
            65: list(range(352, 359)),  # standard + high-dim
        }
        
    elif experiment_type == 'Real - Default':
        # real_world experiment_plan.md — all 9 real datasets, |P|=5,
        # eff-rank 1..5, k subsets per (rank, init). N=100K, scalar.
        # 400 abalone 401 adult 402 autompg 403 concrete 404 drybean
        # 405 hong_kong_weather 406 indiv_household 407 seoul_bike
        # 408 wine_quality_white.
        run_experiment_sets = {
            70: list(range(400, 409)),
        }

    elif experiment_type == 'Synthetic - Default':
        # 100K, 5var, rank 4, LEX, Scalar
        run_experiment_sets = {
            66: list(range(256, 257))
        }
    
    elif experiment_type == 'Synthetic - Rank Sweep':
        run_experiment_sets = {
            67: [253, 254, 255, 256, 257]
        }

    elif experiment_type == 'Synthetic - Flight':
        # Flight 4-var (travel_time, layovers, price, distance), all-MIN.
        run_experiment_sets = {
            68: [267]
        }
    
    elif experiment_type == 'Synthetic - P Sweep':
        run_experiment_sets = {
            68: [248, 265]
        }
    
    elif experiment_type == "Synthetic - Worst Best":
        run_experiment_sets = {
            100: [400, 401]
        }

    else:
        print('Invalid experiment type')
        exit(1)

    # Run the experiments
    config = Bunch()
    config.state = state
    config.run_experiment_sets = run_experiment_sets

    experiment_flow(config)

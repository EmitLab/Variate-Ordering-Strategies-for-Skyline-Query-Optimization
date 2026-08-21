from collections.abc import Callable
from functools import partial

from skylines import n_inits, k
from skylines.dataset import *
from skylines.dataset.real.vos_real_default import get_vos_real_default_datasets
from skylines.dataset.real.celeba import CelebA_40var
from skylines.dataset.real.vos_real_new import (
    NBA_2var_pos, NBA_2var_neg, NBA_3var_mixed, NBA_4var_mixed, NBA_5var_mixed,
    Weather_2var_pos, Weather_2var_neg, Weather_3var_mixed, Weather_4var_mixed, Weather_5var_mixed,
    QWS_2var_pos, QWS_2var_neg, QWS_3var_mixed, QWS_4var_mixed, QWS_5var_mixed,
    CASP_2var_pos, CASP_2var_neg, CASP_3var_mixed, CASP_4var_mixed, CASP_5var_mixed,
    BreastCancer_2var_pos, BreastCancer_2var_neg, BreastCancer_3var_mixed,
    BreastCancer_4var_mixed, BreastCancer_5var_mixed,
    BreastCancer_10var_mixed, BreastCancer_20var_mixed, BreastCancer_30var,
    Parkinsons_2var_pos, Parkinsons_2var_neg, Parkinsons_3var_mixed,
    Parkinsons_4var_mixed, Parkinsons_5var_mixed,
    Pol_2var_pos, Pol_2var_neg, Pol_3var_mixed, Pol_4var_mixed, Pol_5var_mixed,
    Pol_10var_mixed, Pol_26var,
    Elevators_2var_pos, Elevators_2var_neg, Elevators_3var_mixed,
    Elevators_4var_mixed, Elevators_5var_mixed, Elevators_18var,
    SkillCraft_2var_pos, SkillCraft_2var_neg, SkillCraft_3var_mixed,
    SkillCraft_4var_mixed, SkillCraft_5var_mixed,
    KEGG_2var_pos, KEGG_2var_neg, KEGG_3var_mixed, KEGG_4var_mixed, KEGG_5var_mixed,
    KEGG_10var_mixed, KEGG_27var,
)
import numpy as np
from skylines.dataset.synthetic.two_var_var_ordering import PQ_WorstCase_AntiCorr, PQ_BestCase_AntiCorr
from skylines.dataset.synthetic.seven_var_var_ordering import PQRSTUV
from skylines.dataset.synthetic.five_var_var_ordering import PQRST
from skylines.dataset.synthetic.four_var_var_ordering import PQRS
from skylines.dataset.synthetic.three_var_var_ordering import PQR
from skylines.dataset.synthetic.twenty_four_var_var_ordering import TwentyFourVar
from skylines.dataset.synthetic.flight_4var import Flight_4var

def sample_unit_sphere_gaussian(n, rank, seed) -> np.ndarray:
    """
    Sample points uniformly from the surface of the unit sphere in R^dim
    using the Gaussian normalization trick.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, rank))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    return x @ x.T

def recursive_dominance_combinations(preference: list[str], index: int) -> list[dict[str, Dominance]]:
    if index == len(preference) - 1:
        return [{preference[index]: Dominance.MAX}, {preference[index]: Dominance.MIN}]

    next_combinations = recursive_dominance_combinations(preference, index + 1)

    combinations = []
    for combination in next_combinations:
        combinations.append({preference[index]: Dominance.MAX} | combination)
    for combination in next_combinations:
        combinations.append({preference[index]: Dominance.MIN} | combination)

    return combinations


def get_all_dominance_combinations(dataset_class: Dataset.__class__) -> list[
    tuple[Dataset.__class__, dict[str, Dominance]]]:
    dataset: Dataset = dataset_class()
    return [(dataset_class, combination) for combination in recursive_dominance_combinations(sorted(dataset.preference), 0)]

def get_vos_synth_datasets(n_pref_vars: int, rank: int, n_inits: int) -> list[
    tuple[Callable[..., Dataset], dict[str, Dominance]]]:
    datasets: list[tuple[Callable[..., Dataset], dict[str, Dominance]]] = []

    for i in range(n_inits):
        np.random.seed(i)
        
        cov = sample_unit_sphere_gaussian(n=n_pref_vars, rank=rank, seed=i)

        match n_pref_vars:
            case 3:
                dataset_class = partial(PQR, cov_matrix=cov)
            case 4:
                dataset_class = partial(PQRS, cov_matrix=cov)
            case 5:
                dataset_class = partial(PQRST, cov_matrix=cov)
            case 7:
                dataset_class = partial(PQRSTUV, cov_matrix=cov)
            case 24:
                dataset_class = partial(TwentyFourVar, cov_matrix=cov)
            case _:
                raise ValueError(f"Unsupported number of preference variables: {n_pref_vars}")
        
        dataset_class.rank = rank
        dataset_class.init = i
        datasets.append((dataset_class, None))

    return datasets
    
__experiments = None


def get_experiments():
    global __experiments

    if __experiments is None:
        __experiments = {
            
            166: [(PQ_PositiveCorr_1, None)],
            167: [(PQ_PositiveCorr_2, None)],
            168: [(PQ_PositiveCorr_3, None)],
            169: [(PQ_PositiveCorr_4, None)],
            170: [(PQ_PositiveCorr_5, None)],
            171: [(PQ_NegativeCorr_1, None)],
            172: [(PQ_NegativeCorr_2, None)],
            173: [(PQ_NegativeCorr_3, None)],
            174: [(PQ_NegativeCorr_4, None)],
            175: [(PQ_NegativeCorr_5, None)],
            
            176: [(PQR_MixedCorr_1, None)],
            177: [(PQR_MixedCorr_2, None)],
            178: [(PQR_MixedCorr_3, None)],
            179: [(PQR_MixedCorr_4, None)],
            180: [(PQR_MixedCorr_5, None)],
            
            181: [(PQRS_MixedCorr_1, None)],
            182: [(PQRS_MixedCorr_2, None)],
            183: [(PQRS_MixedCorr_3, None)],
            184: [(PQRS_MixedCorr_4, None)],
            185: [(PQRS_MixedCorr_5, None)],
            
            186: [(PQRS_PositiveCorr_1, None)],
            187: [(PQRS_PositiveCorr_2, None)],
            188: [(PQRS_PositiveCorr_3, None)],
            189: [(PQRS_PositiveCorr_4, None)],
            190: [(PQRS_PositiveCorr_5, None)],
            
            191: [(PQRS_NegativeCorr_1, None)],
            192: [(PQRS_NegativeCorr_2, None)],
            193: [(PQRS_NegativeCorr_3, None)],
            194: [(PQRS_NegativeCorr_4, None)],
            195: [(PQRS_NegativeCorr_5, None)],
            
            196: [(PQRS_MixedCorr_6, None)],
            197: [(PQRS_MixedCorr_7, None)],
            198: [(PQRS_MixedCorr_8, None)],
            199: [(PQRS_MixedCorr_9, None)],
            200: [(PQRS_MixedCorr_10, None)],

            201: [(PQRST_MixedCorr_1, None)],
            202: [(PQRST_MixedCorr_2, None)],
            203: [(PQRST_MixedCorr_3, None)],
            204: [(PQRST_MixedCorr_4, None)],
            205: [(PQRST_MixedCorr_5, None)],

            206: [(AutoMPG_2var_pos, None)],
            207: [(Hong_Kong_Weather_2var_pos, None)],
            208: [(DryBean_2var_pos, None)],
            209: [(IndividualHouseholdPowerConsumption_2var_pos, None)],
            210: [(Abalone_2var_pos, None)],
            211: [(ConcreteCompressiveStrength_2var_pos, None)],
            212: [(SeoulBikeDemandDataset_2var_pos, None)],
            213: [(WineQualityWhite_2var_pos, None)],

            214: [(AutoMPG_2var_neg, None)],
            215: [(Hong_Kong_Weather_2var_neg, None)],
            216: [(DryBean_2var_neg, None)],
            217: [(IndividualHouseholdPowerConsumption_2var_neg, None)],
            218: [(Abalone_2var_neg, None)],
            219: [(ConcreteCompressiveStrength_2var_neg, None)],
            220: [(SeoulBikeDemandDataset_2var_neg, None)],
            221: [(WineQualityWhite_2var_neg, None)],

            222: [(AutoMPG_3var_mixed, None)],
            223: [(Hong_Kong_Weather_3var_mixed, None)],
            224: [(DryBean_3var_mixed, None)],
            225: [(IndividualHouseholdPowerConsumption_3var_mixed, None)],
            226: [(Abalone_3var_mixed, None)],
            227: [(ConcreteCompressiveStrength_3var_mixed, None)],
            228: [(SeoulBikeDemandDataset_3var_mixed, None)],
            229: [(WineQualityWhite_3var_mixed, None)],

            230: [(AutoMPG_4var_mixed, None)],
            231: [(Hong_Kong_Weather_4var_mixed, None)],
            232: [(DryBean_4var_mixed, None)],
            233: [(IndividualHouseholdPowerConsumption_4var_mixed, None)],
            234: [(Abalone_4var_mixed, None)],
            235: [(ConcreteCompressiveStrength_4var_mixed, None)],
            236: [(SeoulBikeDemandDataset_4var_mixed, None)],
            237: [(WineQualityWhite_4var_mixed, None)],

            238: [(AutoMPG_5var_mixed, None)],
            239: [(Hong_Kong_Weather_5var_mixed, None)],
            240: [(DryBean_5var_mixed, None)],
            241: [(IndividualHouseholdPowerConsumption_5var_mixed, None)],
            242: [(Abalone_5var_mixed, None)],
            243: [(ConcreteCompressiveStrength_5var_mixed, None)],
            244: [(SeoulBikeDemandDataset_5var_mixed, None)],
            245: [(WineQualityWhite_5var_mixed, None)],
            
            246: get_vos_synth_datasets(n_pref_vars=3, rank=1, n_inits=n_inits),
            247: get_vos_synth_datasets(n_pref_vars=3, rank=2, n_inits=n_inits),
            248: get_vos_synth_datasets(n_pref_vars=3, rank=3, n_inits=n_inits),
            
            249: get_vos_synth_datasets(n_pref_vars=4, rank=1, n_inits=n_inits),
            250: get_vos_synth_datasets(n_pref_vars=4, rank=2, n_inits=n_inits),
            251: get_vos_synth_datasets(n_pref_vars=4, rank=3, n_inits=n_inits),
            252: get_vos_synth_datasets(n_pref_vars=4, rank=4, n_inits=n_inits),
            
            253: get_vos_synth_datasets(n_pref_vars=5, rank=1, n_inits=n_inits),
            254: get_vos_synth_datasets(n_pref_vars=5, rank=2, n_inits=n_inits),
            255: get_vos_synth_datasets(n_pref_vars=5, rank=3, n_inits=n_inits),
            256: get_vos_synth_datasets(n_pref_vars=5, rank=4, n_inits=n_inits),
            257: get_vos_synth_datasets(n_pref_vars=5, rank=5, n_inits=n_inits),
            
            258: get_vos_synth_datasets(n_pref_vars=24, rank=4, n_inits=n_inits),
            259: get_vos_synth_datasets(n_pref_vars=24, rank=20, n_inits=n_inits),
            
            260: get_vos_synth_datasets(n_pref_vars=7, rank=1, n_inits=n_inits),
            261: get_vos_synth_datasets(n_pref_vars=7, rank=2, n_inits=n_inits),
            262: get_vos_synth_datasets(n_pref_vars=7, rank=3, n_inits=n_inits),
            263: get_vos_synth_datasets(n_pref_vars=7, rank=4, n_inits=n_inits),
            264: get_vos_synth_datasets(n_pref_vars=7, rank=5, n_inits=n_inits),
            265: get_vos_synth_datasets(n_pref_vars=7, rank=6, n_inits=n_inits),
            266: get_vos_synth_datasets(n_pref_vars=7, rank=7, n_inits=n_inits),

            # Flight 4-var dataset (travel_time, layovers, price, distance).
            # All-MIN dominance; integer outputs; fixed correlations.
            267: [(Flight_4var, {p: Dominance.MIN for p in
                                  ['travel_time_hours', 'layovers', 'price_usd', 'distance']})],

            # CelebA — 40 facial attribute objectives (all MAX by default)
            300: [(CelebA_40var, None)],

            # ---------------------------------------------------------------
            # New real-world VOS benchmarks (301–358)
            # Run the corresponding prepare_{dataset}.py script before use.
            # ---------------------------------------------------------------

            # NBA Player Statistics (8–15 continuous dims)
            301: [(NBA_2var_pos, None)],
            302: [(NBA_2var_neg, None)],
            303: [(NBA_3var_mixed, None)],
            304: [(NBA_4var_mixed, None)],
            305: [(NBA_5var_mixed, None)],

            # WEATHER — CRU TS climate data via SkyBench (15 dims, 566K rows)
            306: [(Weather_2var_pos, None)],
            307: [(Weather_2var_neg, None)],
            308: [(Weather_3var_mixed, None)],
            309: [(Weather_4var_mixed, None)],
            310: [(Weather_5var_mixed, None)],

            # QWS — Quality of Web Services (9 QoS dims, 2,507 rows)
            311: [(QWS_2var_pos, None)],
            312: [(QWS_2var_neg, None)],
            313: [(QWS_3var_mixed, None)],
            314: [(QWS_4var_mixed, None)],
            315: [(QWS_5var_mixed, None)],

            # CASP — Protein Tertiary Structure (9 features + RMSD, 45,730 rows)
            316: [(CASP_2var_pos, None)],
            317: [(CASP_2var_neg, None)],
            318: [(CASP_3var_mixed, None)],
            319: [(CASP_4var_mixed, None)],
            320: [(CASP_5var_mixed, None)],

            # Breast Cancer Wisconsin (30 dims, 569 rows — high-dim benchmark)
            321: [(BreastCancer_2var_pos, None)],
            322: [(BreastCancer_2var_neg, None)],
            323: [(BreastCancer_3var_mixed, None)],
            324: [(BreastCancer_4var_mixed, None)],
            325: [(BreastCancer_5var_mixed, None)],
            326: [(BreastCancer_10var_mixed, None)],
            327: [(BreastCancer_20var_mixed, None)],
            328: [(BreastCancer_30var, None)],

            # Parkinson's Telemonitoring (16 voice features, 5,875 rows)
            329: [(Parkinsons_2var_pos, None)],
            330: [(Parkinsons_2var_neg, None)],
            331: [(Parkinsons_3var_mixed, None)],
            332: [(Parkinsons_4var_mixed, None)],
            333: [(Parkinsons_5var_mixed, None)],

            # Pol — Robot Arm Control (26 dims, 15,000 rows)
            334: [(Pol_2var_pos, None)],
            335: [(Pol_2var_neg, None)],
            336: [(Pol_3var_mixed, None)],
            337: [(Pol_4var_mixed, None)],
            338: [(Pol_5var_mixed, None)],
            339: [(Pol_10var_mixed, None)],
            340: [(Pol_26var, None)],

            # Elevators — Aircraft Control (18 dims, 16,599 rows)
            341: [(Elevators_2var_pos, None)],
            342: [(Elevators_2var_neg, None)],
            343: [(Elevators_3var_mixed, None)],
            344: [(Elevators_4var_mixed, None)],
            345: [(Elevators_5var_mixed, None)],
            346: [(Elevators_18var, None)],

            # SkillCraft1 — StarCraft 2 gameplay metrics (19 dims, 3,338 rows)
            347: [(SkillCraft_2var_pos, None)],
            348: [(SkillCraft_2var_neg, None)],
            349: [(SkillCraft_3var_mixed, None)],
            350: [(SkillCraft_4var_mixed, None)],
            351: [(SkillCraft_5var_mixed, None)],

            # KEGG Metabolic Network (27 dims, 63,608 rows)
            352: [(KEGG_2var_pos, None)],
            353: [(KEGG_2var_neg, None)],
            354: [(KEGG_3var_mixed, None)],
            355: [(KEGG_4var_mixed, None)],
            356: [(KEGG_5var_mixed, None)],
            357: [(KEGG_10var_mixed, None)],
            358: [(KEGG_27var, None)],

            # ---------------------------------------------------------------
            # Real - Default (real_world experiment_plan.md)
            # |P|=5, effective-rank 1..5. `k` is a PERCENT: per rank r take
            # ceil(k%/100 * |pool_r|) subsets, sampled without replacement,
            # random_state = k_i. Empty rank pools skipped. No init dimension.
            # ---------------------------------------------------------------
            400: get_vos_real_default_datasets('abalone', 5, range(1, 6), k),
            401: get_vos_real_default_datasets('adult', 5, range(1, 6), k),
            402: get_vos_real_default_datasets('autompg', 5, range(1, 6), k),
            403: get_vos_real_default_datasets('concrete_compressive_strength', 5, range(1, 6), k),
            404: get_vos_real_default_datasets('drybean', 5, range(1, 6), k),
            405: get_vos_real_default_datasets('hong_kong_weather', 5, range(1, 6), k),
            406: get_vos_real_default_datasets('individual_household_power_consumption', 5, range(1, 6), k),
            407: get_vos_real_default_datasets('seoul_bike_demand', 5, range(1, 6), k),
            408: get_vos_real_default_datasets('wine_quality_white', 5, range(1, 6), k),
            
            500: [(PQ_WorstCase_AntiCorr, None)],
            501: [(PQ_BestCase_AntiCorr, None)]
        }

    return __experiments

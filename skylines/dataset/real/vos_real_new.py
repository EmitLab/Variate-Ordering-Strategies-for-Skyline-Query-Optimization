"""
VOS benchmark dataset classes for newly integrated real-world datasets.

Each dataset family provides 5 variants with increasing number of preference
variables (2var_pos, 2var_neg, 3var_mixed, 4var_mixed, 5var_mixed) following
the same convention as vos_real.py.

Datasets (data preparation):
    NBA_Players  — skylines/dataset/real/data/NBA_Players/NBA_Players.csv (pre-existing)
    weather      — run data/weather/prepare_weather.py
    qws          — run data/qws/prepare_qws.py
    casp         — run data/casp/prepare_casp.py
    breast_cancer— run data/breast_cancer/prepare_breast_cancer.py
    parkinsons   — run data/parkinsons/prepare_parkinsons.py
    pol          — run data/pol/prepare_pol.py
    elevators    — run data/elevators/prepare_elevators.py
    skillcraft   — run data/skillcraft/prepare_skillcraft.py
    kegg         — run data/kegg/prepare_kegg.py
"""

from skylines.dataset import Dominance
from skylines.dataset.real.real import RealDataset

# ---------------------------------------------------------------------------
# NBA Player Statistics
# Columns: age, player_height, player_weight, draft_round, draft_number,
#          gp, pts, reb, ast, net_rating, oreb_pct, dreb_pct, usg_pct,
#          ts_pct, ast_pct
# ---------------------------------------------------------------------------

_EFFECT_NBA = {
    'usg_pct':    {'pts': 1, 'ast': 1},
    'ast_pct':    {'ast': 1},
    'ts_pct':     {'usg_pct': -1},
    'net_rating': {'pts': 1, 'ast': 1, 'reb': 1},
    'dreb_pct':   {'reb': 1, 'player_height': 1},
    'oreb_pct':   {'reb': 1, 'player_height': 1},
    'reb':        {'player_height': 1, 'player_weight': 1},
    'pts':        {'usg_pct': 1, 'ts_pct': 1},
}


class NBA_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='NBA_Players',
            control=['age', 'player_height', 'player_weight', 'draft_round',
                     'draft_number', 'gp', 'ast', 'net_rating', 'oreb_pct',
                     'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'],
            preference=['pts', 'reb'],
            effect=_EFFECT_NBA,
            dominance=dominance,
            infer_controls={'pts': Dominance.MAX, 'reb': Dominance.MAX},
            size=size, seed=seed, provided=False)


class NBA_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='NBA_Players',
            control=['age', 'player_height', 'player_weight', 'draft_round',
                     'draft_number', 'gp', 'pts', 'reb', 'ast', 'net_rating',
                     'oreb_pct', 'dreb_pct', 'ast_pct'],
            preference=['usg_pct', 'ts_pct'],
            effect=_EFFECT_NBA,
            dominance=dominance,
            infer_controls={'usg_pct': Dominance.MAX, 'ts_pct': Dominance.MAX},
            size=size, seed=seed, provided=False)


class NBA_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='NBA_Players',
            control=['age', 'player_height', 'player_weight', 'draft_round',
                     'draft_number', 'gp', 'net_rating', 'oreb_pct',
                     'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'],
            preference=['pts', 'reb', 'ast'],
            effect=_EFFECT_NBA,
            dominance=dominance,
            infer_controls={'pts': Dominance.MAX, 'reb': Dominance.MAX,
                            'ast': Dominance.MAX},
            size=size, seed=seed, provided=False)


class NBA_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='NBA_Players',
            control=['age', 'player_height', 'player_weight', 'draft_round',
                     'draft_number', 'gp', 'oreb_pct', 'dreb_pct',
                     'usg_pct', 'ast_pct'],
            preference=['pts', 'reb', 'ast', 'net_rating'],
            effect=_EFFECT_NBA,
            dominance=dominance,
            infer_controls={'pts': Dominance.MAX, 'reb': Dominance.MAX,
                            'ast': Dominance.MAX, 'net_rating': Dominance.MAX},
            size=size, seed=seed, provided=False)


class NBA_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='NBA_Players',
            control=['age', 'player_height', 'player_weight', 'draft_round',
                     'draft_number', 'gp', 'oreb_pct', 'dreb_pct', 'ast_pct'],
            preference=['pts', 'reb', 'ast', 'net_rating', 'ts_pct'],
            effect=_EFFECT_NBA,
            dominance=dominance,
            infer_controls={'pts': Dominance.MAX, 'reb': Dominance.MAX,
                            'ast': Dominance.MAX, 'net_rating': Dominance.MAX,
                            'ts_pct': Dominance.MAX},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# WEATHER (CRU TS climate data via SkyBench)
# Columns (15): tmp, tmn, tmx, dtr, pre, cld, vap, wet, frs, pet, rhm, ssh,
#               wnd, stn_pressure, grnd_frost
# ---------------------------------------------------------------------------

_ALL_WEATHER = ['tmp', 'tmn', 'tmx', 'dtr', 'pre', 'cld', 'vap', 'wet',
                'frs', 'pet', 'rhm', 'ssh', 'wnd', 'stn_pressure', 'grnd_frost']


class Weather_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_WEATHER if c not in ('tmp', 'vap')]
        super().__init__(
            file_name='weather',
            control=ctrl,
            preference=['tmp', 'vap'],
            effect={},
            dominance=dominance,
            infer_controls={'tmp': Dominance.MAX, 'vap': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Weather_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_WEATHER if c not in ('tmp', 'pre')]
        super().__init__(
            file_name='weather',
            control=ctrl,
            preference=['tmp', 'pre'],
            effect={},
            dominance=dominance,
            infer_controls={'tmp': Dominance.MAX, 'pre': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Weather_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_WEATHER if c not in ('tmp', 'pre', 'cld')]
        super().__init__(
            file_name='weather',
            control=ctrl,
            preference=['tmp', 'pre', 'cld'],
            effect={},
            dominance=dominance,
            infer_controls={'tmp': Dominance.MAX, 'pre': Dominance.MAX,
                            'cld': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Weather_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_WEATHER if c not in ('tmp', 'pre', 'cld', 'rhm')]
        super().__init__(
            file_name='weather',
            control=ctrl,
            preference=['tmp', 'pre', 'cld', 'rhm'],
            effect={},
            dominance=dominance,
            infer_controls={'tmp': Dominance.MAX, 'pre': Dominance.MAX,
                            'cld': Dominance.MAX, 'rhm': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Weather_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_WEATHER if c not in ('tmp', 'pre', 'cld', 'rhm', 'ssh')]
        super().__init__(
            file_name='weather',
            control=ctrl,
            preference=['tmp', 'pre', 'cld', 'rhm', 'ssh'],
            effect={},
            dominance=dominance,
            infer_controls={'tmp': Dominance.MAX, 'pre': Dominance.MAX,
                            'cld': Dominance.MAX, 'rhm': Dominance.MAX,
                            'ssh': Dominance.MAX},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# QWS — Quality of Web Services
# Columns (9): ResponseTime, Availability, Throughput, Successability,
#              Reliability, Compliance, BestPractices, Latency, Documentation
# ---------------------------------------------------------------------------

_ALL_QWS = ['ResponseTime', 'Availability', 'Throughput', 'Successability',
            'Reliability', 'Compliance', 'BestPractices', 'Latency', 'Documentation']


class QWS_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_QWS if c not in ('Availability', 'Reliability')]
        super().__init__(
            file_name='qws',
            control=ctrl,
            preference=['Availability', 'Reliability'],
            effect={},
            dominance=dominance,
            infer_controls={'Availability': Dominance.MAX, 'Reliability': Dominance.MAX},
            size=size, seed=seed, provided=False)


class QWS_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_QWS if c not in ('Throughput', 'Latency')]
        super().__init__(
            file_name='qws',
            control=ctrl,
            preference=['Throughput', 'Latency'],
            effect={},
            dominance=dominance,
            infer_controls={'Throughput': Dominance.MAX, 'Latency': Dominance.MAX},
            size=size, seed=seed, provided=False)


class QWS_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_QWS if c not in ('Availability', 'Throughput', 'ResponseTime')]
        super().__init__(
            file_name='qws',
            control=ctrl,
            preference=['Availability', 'Throughput', 'ResponseTime'],
            effect={},
            dominance=dominance,
            infer_controls={'Availability': Dominance.MAX, 'Throughput': Dominance.MAX,
                            'ResponseTime': Dominance.MAX},
            size=size, seed=seed, provided=False)


class QWS_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_QWS
                if c not in ('Availability', 'Throughput', 'ResponseTime', 'Reliability')]
        super().__init__(
            file_name='qws',
            control=ctrl,
            preference=['Availability', 'Throughput', 'ResponseTime', 'Reliability'],
            effect={},
            dominance=dominance,
            infer_controls={'Availability': Dominance.MAX, 'Throughput': Dominance.MAX,
                            'ResponseTime': Dominance.MAX, 'Reliability': Dominance.MAX},
            size=size, seed=seed, provided=False)


class QWS_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_QWS
                if c not in ('Availability', 'Throughput', 'ResponseTime',
                             'Reliability', 'Latency')]
        super().__init__(
            file_name='qws',
            control=ctrl,
            preference=['Availability', 'Throughput', 'ResponseTime', 'Reliability', 'Latency'],
            effect={},
            dominance=dominance,
            infer_controls={'Availability': Dominance.MAX, 'Throughput': Dominance.MAX,
                            'ResponseTime': Dominance.MAX, 'Reliability': Dominance.MAX,
                            'Latency': Dominance.MAX},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# CASP — Physicochemical Properties of Protein Tertiary Structure (UCI #265)
# Columns (10): F1-F9 (physicochemical features) + RMSD (target)
# ---------------------------------------------------------------------------

_ALL_CASP = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'RMSD']


class CASP_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_CASP if c not in ('F1', 'F2')]
        super().__init__(
            file_name='casp',
            control=ctrl,
            preference=['F1', 'F2'],
            effect={},
            dominance=dominance,
            infer_controls={'F1': Dominance.MAX, 'F2': Dominance.MAX},
            size=size, seed=seed, provided=False)


class CASP_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_CASP if c not in ('F7', 'F8')]
        super().__init__(
            file_name='casp',
            control=ctrl,
            preference=['F7', 'F8'],
            effect={},
            dominance=dominance,
            infer_controls={'F7': Dominance.MAX, 'F8': Dominance.MAX},
            size=size, seed=seed, provided=False)


class CASP_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_CASP if c not in ('RMSD', 'F1', 'F7')]
        super().__init__(
            file_name='casp',
            control=ctrl,
            preference=['RMSD', 'F1', 'F7'],
            effect={},
            dominance=dominance,
            infer_controls={'RMSD': Dominance.MAX, 'F1': Dominance.MAX,
                            'F7': Dominance.MAX},
            size=size, seed=seed, provided=False)


class CASP_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_CASP if c not in ('RMSD', 'F1', 'F2', 'F7')]
        super().__init__(
            file_name='casp',
            control=ctrl,
            preference=['RMSD', 'F1', 'F2', 'F7'],
            effect={},
            dominance=dominance,
            infer_controls={'RMSD': Dominance.MAX, 'F1': Dominance.MAX,
                            'F2': Dominance.MAX, 'F7': Dominance.MAX},
            size=size, seed=seed, provided=False)


class CASP_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_CASP if c not in ('RMSD', 'F1', 'F2', 'F7', 'F8')]
        super().__init__(
            file_name='casp',
            control=ctrl,
            preference=['RMSD', 'F1', 'F2', 'F7', 'F8'],
            effect={},
            dominance=dominance,
            infer_controls={'RMSD': Dominance.MAX, 'F1': Dominance.MAX,
                            'F2': Dominance.MAX, 'F7': Dominance.MAX,
                            'F8': Dominance.MAX},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# Breast Cancer Wisconsin (Diagnostic) — UCI #17 / sklearn
# 30 continuous features — good high-dimensional benchmark
# Columns: mean_{b}, se_{b}, worst_{b} for b in 10 nuclear image features
# ---------------------------------------------------------------------------

_BC_BASE = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
_ALL_BC = (
    [f'mean_{b}' for b in _BC_BASE] +
    [f'se_{b}'   for b in _BC_BASE] +
    [f'worst_{b}' for b in _BC_BASE]
)


class BreastCancer_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_BC if c not in ('mean_radius', 'mean_area')]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=['mean_radius', 'mean_area'],
            effect={},
            dominance=dominance,
            infer_controls={'mean_radius': Dominance.MAX, 'mean_area': Dominance.MAX},
            size=size, seed=seed, provided=False)


class BreastCancer_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_BC if c not in ('mean_smoothness', 'mean_concavity')]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=['mean_smoothness', 'mean_concavity'],
            effect={},
            dominance=dominance,
            infer_controls={'mean_smoothness': Dominance.MAX, 'mean_concavity': Dominance.MAX},
            size=size, seed=seed, provided=False)


class BreastCancer_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['mean_radius', 'mean_texture', 'mean_smoothness']
        ctrl = [c for c in _ALL_BC if c not in pref]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class BreastCancer_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['mean_radius', 'mean_texture', 'mean_smoothness', 'mean_compactness']
        ctrl = [c for c in _ALL_BC if c not in pref]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class BreastCancer_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['mean_radius', 'mean_texture', 'mean_smoothness',
                'mean_compactness', 'mean_concavity']
        ctrl = [c for c in _ALL_BC if c not in pref]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class BreastCancer_10var_mixed(RealDataset):
    """All 10 mean features as preferences."""
    def __init__(self, dominance=None, size=None, seed=42):
        pref = [f'mean_{b}' for b in _BC_BASE]
        ctrl = [c for c in _ALL_BC if c not in pref]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class BreastCancer_20var_mixed(RealDataset):
    """Mean + SE features (20 dims) as preferences."""
    def __init__(self, dominance=None, size=None, seed=42):
        pref = [f'mean_{b}' for b in _BC_BASE] + [f'se_{b}' for b in _BC_BASE]
        ctrl = [c for c in _ALL_BC if c not in pref]
        super().__init__(
            file_name='breast_cancer',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class BreastCancer_30var(RealDataset):
    """All 30 features as preferences — full high-dimensional benchmark."""
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='breast_cancer',
            control=[],
            preference=_ALL_BC,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in _ALL_BC},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# Parkinson's Telemonitoring — UCI #189
# 16 voice biomarker features + 2 UPDRS targets + demographics
# Columns: age, sex, test_time, motor_updrs, total_updrs,
#          jitter_pct, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp,
#          shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11,
#          shimmer_dda, nhr, hnr, rpde, dfa, ppe
# ---------------------------------------------------------------------------

_ALL_PARK = ['age', 'sex', 'test_time', 'motor_updrs', 'total_updrs',
             'jitter_pct', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
             'shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11',
             'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']


class Parkinsons_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_PARK if c not in ('jitter_pct', 'shimmer')]
        super().__init__(
            file_name='parkinsons',
            control=ctrl,
            preference=['jitter_pct', 'shimmer'],
            effect={},
            dominance=dominance,
            infer_controls={'jitter_pct': Dominance.MAX, 'shimmer': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Parkinsons_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_PARK if c not in ('hnr', 'nhr')]
        super().__init__(
            file_name='parkinsons',
            control=ctrl,
            preference=['hnr', 'nhr'],
            effect={},
            dominance=dominance,
            infer_controls={'hnr': Dominance.MAX, 'nhr': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Parkinsons_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['jitter_pct', 'shimmer', 'hnr']
        ctrl = [c for c in _ALL_PARK if c not in pref]
        super().__init__(
            file_name='parkinsons',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Parkinsons_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['jitter_pct', 'shimmer', 'hnr', 'rpde']
        ctrl = [c for c in _ALL_PARK if c not in pref]
        super().__init__(
            file_name='parkinsons',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Parkinsons_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['jitter_pct', 'shimmer', 'hnr', 'rpde', 'dfa']
        ctrl = [c for c in _ALL_PARK if c not in pref]
        super().__init__(
            file_name='parkinsons',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# Pol — Robot Arm Control (OpenML #60 / Delve)
# 26 continuous inputs + 1 continuous target
# Columns: pol_f00 .. pol_f25, pol_y
# ---------------------------------------------------------------------------

_ALL_POL = [f'pol_f{i:02d}' for i in range(26)] + ['pol_y']


class Pol_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_POL if c not in ('pol_f00', 'pol_f01')]
        super().__init__(
            file_name='pol',
            control=ctrl,
            preference=['pol_f00', 'pol_f01'],
            effect={},
            dominance=dominance,
            infer_controls={'pol_f00': Dominance.MAX, 'pol_f01': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Pol_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_POL if c not in ('pol_f00', 'pol_f13')]
        super().__init__(
            file_name='pol',
            control=ctrl,
            preference=['pol_f00', 'pol_f13'],
            effect={},
            dominance=dominance,
            infer_controls={'pol_f00': Dominance.MAX, 'pol_f13': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Pol_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['pol_f00', 'pol_f01', 'pol_f13']
        ctrl = [c for c in _ALL_POL if c not in pref]
        super().__init__(
            file_name='pol',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Pol_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['pol_f00', 'pol_f01', 'pol_f13', 'pol_f14']
        ctrl = [c for c in _ALL_POL if c not in pref]
        super().__init__(
            file_name='pol',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Pol_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['pol_f00', 'pol_f01', 'pol_f13', 'pol_f14', 'pol_y']
        ctrl = [c for c in _ALL_POL if c not in pref]
        super().__init__(
            file_name='pol',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Pol_10var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = [f'pol_f{i:02d}' for i in range(10)]
        ctrl = [c for c in _ALL_POL if c not in pref]
        super().__init__(
            file_name='pol',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Pol_26var(RealDataset):
    """All 26 input features as preferences (no target)."""
    def __init__(self, dominance=None, size=None, seed=42):
        pref = [f'pol_f{i:02d}' for i in range(26)]
        super().__init__(
            file_name='pol',
            control=['pol_y'],
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# Elevators — Aircraft Control (OpenML #216 / Delve)
# 18 continuous inputs + 1 continuous target
# Columns: elev_f00 .. elev_f17, elev_y
# ---------------------------------------------------------------------------

_ALL_ELEV = [f'elev_f{i:02d}' for i in range(18)] + ['elev_y']


class Elevators_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_ELEV if c not in ('elev_f00', 'elev_f01')]
        super().__init__(
            file_name='elevators',
            control=ctrl,
            preference=['elev_f00', 'elev_f01'],
            effect={},
            dominance=dominance,
            infer_controls={'elev_f00': Dominance.MAX, 'elev_f01': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Elevators_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_ELEV if c not in ('elev_f00', 'elev_f09')]
        super().__init__(
            file_name='elevators',
            control=ctrl,
            preference=['elev_f00', 'elev_f09'],
            effect={},
            dominance=dominance,
            infer_controls={'elev_f00': Dominance.MAX, 'elev_f09': Dominance.MAX},
            size=size, seed=seed, provided=False)


class Elevators_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['elev_f00', 'elev_f01', 'elev_f09']
        ctrl = [c for c in _ALL_ELEV if c not in pref]
        super().__init__(
            file_name='elevators',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Elevators_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['elev_f00', 'elev_f01', 'elev_f09', 'elev_f10']
        ctrl = [c for c in _ALL_ELEV if c not in pref]
        super().__init__(
            file_name='elevators',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Elevators_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['elev_f00', 'elev_f01', 'elev_f09', 'elev_f10', 'elev_y']
        ctrl = [c for c in _ALL_ELEV if c not in pref]
        super().__init__(
            file_name='elevators',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class Elevators_18var(RealDataset):
    """All 18 input features as preferences (no target)."""
    def __init__(self, dominance=None, size=None, seed=42):
        pref = [f'elev_f{i:02d}' for i in range(18)]
        super().__init__(
            file_name='elevators',
            control=['elev_y'],
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# SkillCraft1 — StarCraft 2 Gameplay Metrics (UCI #272)
# 19 gameplay metrics (GameID dropped)
# Columns: LeagueIndex, Age, HoursPerWeek, TotalHours, APM,
#          SelectByHotkeys, AssignToHotkeys, UniqueHotkeys,
#          MinimapAttacks, MinimapRightClicks, NumberOfPACs,
#          GapBetweenPACs, ActionLatency, ActionsInPAC,
#          TotalMapExplored, WorkersMade, UniqueUnitsMade,
#          ComplexUnitsMade, ComplexAbilitiesUsed
# ---------------------------------------------------------------------------

_ALL_SC = ['LeagueIndex', 'Age', 'HoursPerWeek', 'TotalHours', 'APM',
           'SelectByHotkeys', 'AssignToHotkeys', 'UniqueHotkeys',
           'MinimapAttacks', 'MinimapRightClicks', 'NumberOfPACs',
           'GapBetweenPACs', 'ActionLatency', 'ActionsInPAC',
           'TotalMapExplored', 'WorkersMade', 'UniqueUnitsMade',
           'ComplexUnitsMade', 'ComplexAbilitiesUsed']

_EFFECT_SC = {
    'APM':            {'TotalHours': 1, 'HoursPerWeek': 1},
    'SelectByHotkeys':{'APM': 1},
    'ActionLatency':  {'APM': -1},
    'LeagueIndex':    {'APM': 1, 'TotalHours': 1},
}


class SkillCraft_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_SC if c not in ('TotalHours', 'APM')]
        super().__init__(
            file_name='skillcraft',
            control=ctrl,
            preference=['TotalHours', 'APM'],
            effect=_EFFECT_SC,
            dominance=dominance,
            infer_controls={'TotalHours': Dominance.MAX, 'APM': Dominance.MAX},
            size=size, seed=seed, provided=False)


class SkillCraft_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_SC if c not in ('APM', 'ActionLatency')]
        super().__init__(
            file_name='skillcraft',
            control=ctrl,
            preference=['APM', 'ActionLatency'],
            effect=_EFFECT_SC,
            dominance=dominance,
            infer_controls={'APM': Dominance.MAX, 'ActionLatency': Dominance.MAX},
            size=size, seed=seed, provided=False)


class SkillCraft_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['APM', 'ActionLatency', 'TotalMapExplored']
        ctrl = [c for c in _ALL_SC if c not in pref]
        super().__init__(
            file_name='skillcraft',
            control=ctrl,
            preference=pref,
            effect=_EFFECT_SC,
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class SkillCraft_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['APM', 'ActionLatency', 'TotalMapExplored', 'UniqueHotkeys']
        ctrl = [c for c in _ALL_SC if c not in pref]
        super().__init__(
            file_name='skillcraft',
            control=ctrl,
            preference=pref,
            effect=_EFFECT_SC,
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class SkillCraft_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['APM', 'ActionLatency', 'TotalMapExplored', 'UniqueHotkeys', 'WorkersMade']
        ctrl = [c for c in _ALL_SC if c not in pref]
        super().__init__(
            file_name='skillcraft',
            control=ctrl,
            preference=pref,
            effect=_EFFECT_SC,
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


# ---------------------------------------------------------------------------
# KEGG Metabolic Reaction Network (Undirected) — UCI #221
# 27 continuous network topology features, 63,608 records
# Columns: kegg_f00 .. kegg_f26
# ---------------------------------------------------------------------------

_ALL_KEGG = [f'kegg_f{i:02d}' for i in range(27)]


class KEGG_2var_pos(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_KEGG if c not in ('kegg_f00', 'kegg_f01')]
        super().__init__(
            file_name='kegg',
            control=ctrl,
            preference=['kegg_f00', 'kegg_f01'],
            effect={},
            dominance=dominance,
            infer_controls={'kegg_f00': Dominance.MAX, 'kegg_f01': Dominance.MAX},
            size=size, seed=seed, provided=False)


class KEGG_2var_neg(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        ctrl = [c for c in _ALL_KEGG if c not in ('kegg_f00', 'kegg_f13')]
        super().__init__(
            file_name='kegg',
            control=ctrl,
            preference=['kegg_f00', 'kegg_f13'],
            effect={},
            dominance=dominance,
            infer_controls={'kegg_f00': Dominance.MAX, 'kegg_f13': Dominance.MAX},
            size=size, seed=seed, provided=False)


class KEGG_3var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['kegg_f00', 'kegg_f01', 'kegg_f13']
        ctrl = [c for c in _ALL_KEGG if c not in pref]
        super().__init__(
            file_name='kegg',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class KEGG_4var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['kegg_f00', 'kegg_f01', 'kegg_f13', 'kegg_f14']
        ctrl = [c for c in _ALL_KEGG if c not in pref]
        super().__init__(
            file_name='kegg',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class KEGG_5var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = ['kegg_f00', 'kegg_f01', 'kegg_f13', 'kegg_f14', 'kegg_f26']
        ctrl = [c for c in _ALL_KEGG if c not in pref]
        super().__init__(
            file_name='kegg',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class KEGG_10var_mixed(RealDataset):
    def __init__(self, dominance=None, size=None, seed=42):
        pref = [f'kegg_f{i:02d}' for i in range(10)]
        ctrl = [c for c in _ALL_KEGG if c not in pref]
        super().__init__(
            file_name='kegg',
            control=ctrl,
            preference=pref,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in pref},
            size=size, seed=seed, provided=False)


class KEGG_27var(RealDataset):
    """All 27 network topology features as preferences."""
    def __init__(self, dominance=None, size=None, seed=42):
        super().__init__(
            file_name='kegg',
            control=[],
            preference=_ALL_KEGG,
            effect={},
            dominance=dominance,
            infer_controls={p: Dominance.MAX for p in _ALL_KEGG},
            size=size, seed=seed, provided=False)

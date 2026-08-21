from .dataset import Dataset, Dominance
from .real.real import RealDataset

from .real.vos_real import *
from .real.vos_real_new import *
from .real.celeba import CelebA_40var

from .synthetic.two_var_var_ordering import PQ_PositiveCorr_1, PQ_PositiveCorr_2, PQ_PositiveCorr_3, PQ_PositiveCorr_4, PQ_PositiveCorr_5, PQ_NegativeCorr_1, PQ_NegativeCorr_2, PQ_NegativeCorr_3, PQ_NegativeCorr_4, PQ_NegativeCorr_5, PQ_WorstCase_AntiCorr, PQ_BestCase_AntiCorr
from .synthetic.three_var_var_ordering import PQR_MixedCorr_1, PQR_MixedCorr_2, PQR_MixedCorr_3, PQR_MixedCorr_4, PQR_MixedCorr_5
from .synthetic.four_var_var_ordering import PQRS_MixedCorr_1, PQRS_MixedCorr_2, PQRS_MixedCorr_3, PQRS_MixedCorr_4, PQRS_MixedCorr_5, PQRS_MixedCorr_6, PQRS_MixedCorr_7, PQRS_MixedCorr_8, PQRS_MixedCorr_9, PQRS_MixedCorr_10, PQRS_PositiveCorr_1, PQRS_PositiveCorr_2, PQRS_PositiveCorr_3, PQRS_PositiveCorr_4, PQRS_PositiveCorr_5, PQRS_NegativeCorr_1, PQRS_NegativeCorr_2, PQRS_NegativeCorr_3, PQRS_NegativeCorr_4, PQRS_NegativeCorr_5
from .synthetic.five_var_var_ordering import PQRST_MixedCorr_1, PQRST_MixedCorr_2, PQRST_MixedCorr_3, PQRST_MixedCorr_4, PQRST_MixedCorr_5
from .synthetic.flight_4var import Flight_4var

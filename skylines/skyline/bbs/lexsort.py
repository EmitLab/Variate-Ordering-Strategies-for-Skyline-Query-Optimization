from skylines.dataset import Dataset
from skylines.skyline.bbs.bbs import BBSSkyline


class BBSSkylineLexsort(BBSSkyline):
    """BBS with a stable np.lexsort(-data.T[::-1]) presort of the leaf
    entries before STR bulk-load. The presort runs inside the C kernel
    (skyline_bbs_lex_run), mirroring BNL/SFS/SaLSa/D&C lexsort variants."""

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._lex_sort = True

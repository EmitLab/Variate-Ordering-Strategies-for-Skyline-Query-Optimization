import os
from pathlib import Path

import numpy as np
import pandas as pd

from skylines.common.utils import safe_str, format_num
from skylines.dataset import Dominance
from skylines.dataset.real.real import RealDataset

# All 40 CelebA facial attribute names exactly as they appear in the CSV header.
# CelebA encodes each attribute as -1 (absent) or +1 (present).
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
    'Young',
]

_DATA_DIR = Path(__file__).parent.resolve() / 'data' / 'celeba'


class CelebA_40var(RealDataset):
    """
    CelebA — 40 facial attribute objectives.

    Each of the 202,599 CelebA images is annotated with 40 binary attributes
    (encoded as -1/+1 in the source file).  This dataset exposes all 40
    attributes as MAX-preference objectives and no control variables.

    Data preparation:
        Run  skylines/dataset/real/data/celeba/prepare_celeba.py  to download
        and convert the official annotation file to celeba.csv.

    Dominance:
        All attributes default to Dominance.MAX.  Pass a custom `dominance`
        dict to override individual directions.
    """

    def __init__(
        self,
        dominance: dict[str, Dominance] = None,
        size: int = None,
        seed: int = 42,
    ):
        super().__init__(
            file_name='celeba',
            control=[],
            preference=CELEBA_ATTRIBUTES,
            effect={},           # no assumed causal structure
            dominance=dominance,
            infer_controls=None,
            size=size,
            seed=seed,
            provided=False,
        )

    # ------------------------------------------------------------------
    # Override _load() for three reasons specific to binary attribute data:
    #
    #  1. CelebA uses -1/+1 encoding; remap to 0/1 so Gaussian noise
    #     creates meaningful continuous spread rather than values that
    #     straddle ±1.
    #
    #  2. The parent's integer-rounding heuristic (df[col].is_integer())
    #     would collapse all noisy values back to 0 or 1, defeating the
    #     purpose of the noise.  We skip that step entirely.
    #
    #  3. CelebA has 202,599 rows — larger than the default n_samples
    #     (50 K), so we sample without replacement when possible.
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        csv_path = _DATA_DIR / 'celeba.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CelebA attribute file not found at {csv_path}\n"
                f"Run  python {_DATA_DIR / 'prepare_celeba.py'}  to download it."
            )

        df = pd.read_csv(csv_path)
        df.columns = [safe_str(col) for col in df.columns]
        df = df[self.variates]
        df = df.apply(pd.to_numeric, errors='coerce')

        if self.provided:
            provided_path = _DATA_DIR / f'celeba_{format_num(self.size)}_{self.seed}.csv'
            df = pd.read_csv(provided_path)
            df.columns = [safe_str(col) for col in df.columns]
            df = df[self.variates].apply(pd.to_numeric, errors='coerce')
            return df.fillna(df.max() + df.std())

        # Remap -1/+1 → 0/1 before adding noise so that:
        #   • std is measured on [0,1]-range data (≈ 0.5 for balanced attrs)
        #   • noise stays within a natural [0,1] domain after clipping
        df = (df + 1.0) / 2.0

        np.random.seed(self.seed)
        replace = len(df) < self.size
        post_df = df.sample(n=self.size, replace=replace).reset_index(drop=True)

        for column in df.columns:
            scale = df[column].std() / 6.0   # matches parent heuristic
            noise = np.random.normal(loc=0.0, scale=scale, size=self.size)
            post_df[column] = (post_df[column] + noise).clip(0.0, 1.0)

        return post_df.fillna(post_df.max() + post_df.std())

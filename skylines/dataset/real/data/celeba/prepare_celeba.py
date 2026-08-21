#!/usr/bin/env python3
"""
Download and prepare the CelebA attribute data for Causal-Skylines.

Produces:  skylines/dataset/real/data/celeba/celeba.csv
Columns:   image_id (str) + all 40 attribute columns (int: -1 or 1)

Usage
-----
    # Auto-download via gdown (requires: pip install gdown):
    python prepare_celeba.py

    # Convert a file you already have (list_attr_celeba.txt or .csv):
    python prepare_celeba.py --source /path/to/list_attr_celeba.txt
    python prepare_celeba.py --source /path/to/list_attr_celeba.csv

Accepted source formats
-----------------------
1. Official text format (list_attr_celeba.txt):
       202599
       5_o_Clock_Shadow Arched_Eyebrows ...
       000001.jpg -1  1  1 ...

2. Kaggle CSV (list_attr_celeba.csv):
       image_id,5_o_Clock_Shadow,Arched_Eyebrows,...
       000001.jpg,-1,1,1,...
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).parent.resolve()
_OUT_PATH = _THIS_DIR / 'celeba.csv'

# Official Google Drive file ID for list_attr_celeba.txt
_GDRIVE_FILE_ID = '0B7EVK8r0v71pblRyaVFSWGxPY0U'

_ATTRIBUTES = [
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


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_txt(path: Path) -> pd.DataFrame:
    """Parse the official list_attr_celeba.txt format."""
    with open(path, 'r') as fh:
        n_rows = int(fh.readline().strip())
        header = fh.readline().strip().split()
        rows = []
        for line in fh:
            parts = line.strip().split()
            rows.append(parts)

    df = pd.DataFrame(rows, columns=['image_id'] + header)
    for col in header:
        df[col] = pd.to_numeric(df[col])
    print(f"  Parsed {len(df):,} rows from text format  ({path.name})")
    return df


def _parse_csv(path: Path) -> pd.DataFrame:
    """Parse a CSV that already has a header row (e.g. Kaggle download)."""
    df = pd.read_csv(path)
    print(f"  Parsed {len(df):,} rows from CSV  ({path.name})")
    return df


def _load_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.txt':
        return _parse_txt(path)
    if suffix in {'.csv', '.tsv'}:
        return _parse_csv(path)
    # Sniff by content
    with open(path, 'r') as fh:
        first = fh.readline().strip()
    if first.isdigit():
        return _parse_txt(path)
    return _parse_csv(path)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _try_gdown(dest: Path) -> bool:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("  gdown not installed — skipping (pip install gdown to enable).")
        return False

    url = f'https://drive.google.com/uc?id={_GDRIVE_FILE_ID}'
    print(f"  Downloading via gdown  (file id: {_GDRIVE_FILE_ID}) ...")
    try:
        import gdown
        gdown.download(url, str(dest), quiet=False)
        return dest.exists()
    except Exception as exc:
        print(f"  gdown failed: {exc}")
        return False


def _try_torchvision(dest_dir: Path) -> bool:
    try:
        import torchvision  # noqa: F401
    except ImportError:
        print("  torchvision not installed — skipping.")
        return False

    try:
        import torchvision.datasets as tvd
        print("  Downloading CelebA via torchvision (this may take a while) ...")
        tvd.CelebA(root=str(dest_dir / '_tv_cache'), split='all',
                   target_type='attr', download=True)
        # torchvision stores the txt at celeba/Anno/list_attr_celeba.txt
        txt = dest_dir / '_tv_cache' / 'celeba' / 'Anno' / 'list_attr_celeba.txt'
        if txt.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(txt, dest_dir / 'list_attr_celeba.txt')
            return True
        return False
    except Exception as exc:
        print(f"  torchvision download failed: {exc}")
        return False


def _print_manual_instructions():
    print("""
Manual download instructions
----------------------------
Option A — Kaggle:
  1. Install the Kaggle CLI:       pip install kaggle
  2. Set up your API key:          https://www.kaggle.com/docs/api
  3. Download the attribute file:
       kaggle datasets download jessicali9530/celeba-dataset \\
           -f list_attr_celeba.csv --path .
  4. Re-run this script:
       python prepare_celeba.py --source list_attr_celeba.csv

Option B — Official site:
  1. Visit https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  2. Download  Anno/list_attr_celeba.txt  from Google Drive.
  3. Re-run this script:
       python prepare_celeba.py --source list_attr_celeba.txt

Option C — Hugging Face (requires datasets library):
  pip install datasets
  python - <<'EOF'
  from datasets import load_dataset
  import pandas as pd
  ds = load_dataset("flax-sentence-embeddings/celeba_attrs", split="train")
  pd.DataFrame(ds).to_csv("list_attr_celeba.csv", index=False)
  EOF
  python prepare_celeba.py --source list_attr_celeba.csv
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(source: Path | None):
    if _OUT_PATH.exists():
        print(f"celeba.csv already exists at {_OUT_PATH} — nothing to do.")
        print("Delete it and re-run to regenerate.")
        return

    if source is None:
        # Try auto-download strategies in order
        print("No source file specified — attempting automatic download ...")

        txt_dest = _THIS_DIR / 'list_attr_celeba.txt'

        if _try_gdown(txt_dest):
            source = txt_dest
        elif _try_torchvision(_THIS_DIR):
            source = _THIS_DIR / 'list_attr_celeba.txt'
        else:
            print("\nAutomatic download failed.")
            _print_manual_instructions()
            sys.exit(1)

    # Load from source
    print(f"Loading source: {source}")
    df = _load_source(Path(source))

    # Validate columns
    missing = [a for a in _ATTRIBUTES if a not in df.columns]
    if missing:
        raise ValueError(
            f"Source file is missing {len(missing)} expected attribute columns: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep image_id + all 40 attributes
    keep_cols = ['image_id'] if 'image_id' in df.columns else []
    df = df[keep_cols + _ATTRIBUTES]

    # Ensure integer encoding (-1/1)
    for col in _ATTRIBUTES:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int8')

    _THIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows × {len(_ATTRIBUTES)} attributes → {_OUT_PATH}")
    print("Dataset is ready for use with CelebA_40var.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=None,
                        help='Path to list_attr_celeba.txt or .csv (optional; '
                             'auto-downloaded if omitted)')
    args = parser.parse_args()
    prepare(args.source)

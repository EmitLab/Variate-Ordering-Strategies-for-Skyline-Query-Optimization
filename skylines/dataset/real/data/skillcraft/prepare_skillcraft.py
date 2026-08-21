"""
Download and prepare the SkillCraft1 Master Table Dataset.

Source: UCI ML Repository #272
        19 gameplay metrics from StarCraft 2 replays, 3,338 records.
        https://archive.ics.uci.edu/dataset/272

Usage:
    python prepare_skillcraft.py

Output: skillcraft.csv in this directory (GameID dropped).

Columns:
    LeagueIndex, Age, HoursPerWeek, TotalHours, APM,
    SelectByHotkeys, AssignToHotkeys, UniqueHotkeys,
    MinimapAttacks, MinimapRightClicks, NumberOfPACs,
    GapBetweenPACs, ActionLatency, ActionsInPAC,
    TotalMapExplored, WorkersMade, UniqueUnitsMade,
    ComplexUnitsMade, ComplexAbilitiesUsed
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'skillcraft.csv'

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv'


def main():
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        return

    try:
        import requests
        import pandas as pd
        from io import StringIO
    except ImportError as e:
        print(f'Missing dependency: {e}. Run: pip install requests pandas')
        sys.exit(1)

    print(f'Downloading SkillCraft1 from {URL} …')
    r = requests.get(URL, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))

    # Drop GameID (identifier)
    for col in ['GameID', 'game_id']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()

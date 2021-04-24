from pathlib import Path

import pandas as pd

from app.columns import Emotion, Column, Genre


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def calc_track_hash(album_id, track_id, hash_suffix) -> str:
    return f'{album_id}:{track_id}:{hash_suffix}'


def filter_has_emotion(df: pd.DataFrame, emotion: Emotion) -> pd.DataFrame:
    return df.loc[df.emotion == emotion.value]


def filter_has_genre(df: pd.DataFrame, genre: Genre) -> pd.DataFrame:
    return df.loc[df.genre == genre.value]


def create_empty_frame() -> pd.DataFrame:
    return pd.DataFrame(index=[], columns=list(map(lambda column: column.value, Column)))


def ensure_dirs_exist(dir_paths: list[Path]):
    for path in dir_paths:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

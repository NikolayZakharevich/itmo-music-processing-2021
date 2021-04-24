import re

import pandas as pd

from app.columns import map_genre_raw, Column, Emotion, AssignmentColumn, Genre
from app.utils import calc_track_hash
from config import STATIC_STORAGE_URL


def df_filter_has_genre(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.genre.notnull()]


def df_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=[Column.YM_TRACK_ID.value, Column.YM_ALBUM_ID.value])


def df_filter_unpopular_genres(df: pd.DataFrame) -> pd.DataFrame:
    unpopular_genres = [Genre.BLUES.value, Genre.COUNTRY.value, Genre.LOUNGE.value, Genre.PUNK.value]
    return df[df.genre.isin(unpopular_genres)]


def df_drop_narrow_genres(df: pd.DataFrame) -> pd.DataFrame:
    pd.options.mode.chained_assignment = None

    genres_to_drop = ['african', 'bollywood', 'caucasian', 'forchildren', 'eastern', 'trance', 'sport',
                      'shanson', 'naturesounds', 'jewish', 'balkan', 'children', 'georgian', 'celtic',
                      'musical', 'funk', 'vocal', 'dnb', 'experimental', 'animated', 'reggae', 'industrial',
                      'relax']
    df = df[(~df.genre.isin(genres_to_drop)) & df.genre.notnull()]
    df[Column.GENRE.value] = df.genre.apply(lambda g: map_genre_raw(g).value)
    return df


def df_reduce_popular_genres(df: pd.DataFrame) -> pd.DataFrame:
    return df[((df.genre != Genre.POP.value) | (df.ym_track_id % 10 <= 1))
              & ((df.genre != Genre.ROCK.value) | (df.ym_track_id % 10 <= 2))
              & ((df.genre != Genre.ELECTRONIC.value) | (df.ym_track_id % 10 <= 4))
              & ((df.genre != Genre.ALTERNATIVE.value) | (df.ym_track_id % 10 <= 5))
              & ((df.genre != Genre.INDIE.value) | (df.ym_track_id % 10 <= 5))
              & ((df.genre != Genre.RAP.value) | (df.ym_track_id % 10 <= 6))
              & ((df.genre != Genre.SOUNDTRACK.value) | (df.ym_track_id % 10 <= 7))
              & ((df.genre != Genre.DANCE.value) | (df.ym_track_id % 10 <= 8))
              ]


def df_reduce_popular_emotions(df: pd.DataFrame) -> pd.DataFrame:
    return df[((df.emotion != Emotion.QUIET.value) | (df.ym_track_id % 10 <= 1))
              & ((df.emotion != Emotion.NOSTALGIC.value) | (df.ym_track_id % 10 <= 2))
              & ((df.emotion != Emotion.SADNESS.value) | (df.ym_track_id % 10 <= 2))
              & ((df.emotion != Emotion.SWEET.value) | (df.ym_track_id % 10 <= 4))
              & ((df.emotion != Emotion.SURPRISE.value) | (df.ym_track_id % 10 <= 4))
              & ((df.emotion != Emotion.SOULFUL.value) | (df.ym_track_id % 10 <= 5))
              & ((df.emotion != Emotion.ROMANTIC.value) | (df.ym_track_id % 10 <= 5))
              & ((df.emotion != Emotion.LONELY.value) | (df.ym_track_id % 10 <= 6))
              & ((df.emotion != Emotion.COMFORTABLE.value) | (df.ym_track_id % 10 <= 6))
              ]

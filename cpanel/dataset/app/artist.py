from pathlib import Path
from time import sleep
from typing import Union, Any

import pandas as pd
import requests
import urllib.request
import lxml.etree as ET
import os

from urllib.parse import quote_plus
from app.columns import Column
from app.utils import ensure_dirs_exist
from config import FILE_ARTISTS_INFO, JAMENDO_CLIENT_ID

REQUESTS_LIMIT = 1
CACHE_COLUMN_REQUESTS_CNT = 'requests_cnt'

CACHE_PATH = FILE_ARTISTS_INFO
CACHE_COLUMNS = [Column.ARTIST_NAME.value, Column.ARTIST_COUNTRY.value, Column.ARTIST_SEX.value,
                 Column.ARTIST_YEAR.value, CACHE_COLUMN_REQUESTS_CNT]

URL_MUSICOVERY_GET_INFO = 'https://musicovery.com/api/V5/artist.php?fct=getinfo&id=%s'
URL_MUSICOVERY_SEARCH = 'https://musicovery.com/api/V5/artist.php?fct=search&artistname=%s'

URL_JAMENDO_SEARCH = f'https://api.jamendo.com/v3.0/artists/locations/?client_id={JAMENDO_CLIENT_ID}&namesearch=%s'

FIELD_MUSICOVERY_CODE = 'code'
FIELD_MUSICOVERY_ANSWER = 'answer'
FIELD_MUSICOVERY_COUNTRY = 'country'
FIELD_MUSICOVERY_MBID = 'mbid'
FIELD_MUSICOVERY_SEXE = 'sexe'
FIELD_MUSICOVERY_YEAR_DEBUT = 'year_debut'

FIELD_JAMENDO_RESULTS = 'results'
FIELD_JAMENDO_LOCATIONS = 'locations'
FIELD_JAMENDO_COUNTRY = 'country'


class ArtistInfo():
    musicovery_limit_reached = False
    cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def update_artists_info_from_cache(cls, tracks: pd.DataFrame) -> pd.DataFrame:
        cls._load_cache()
        updated_tracks = []
        for i, track in tracks.iterrows():
            if not pd.isna(track[Column.ARTIST_COUNTRY.value]) and \
                    not pd.isna(track[Column.ARTIST_SEX.value]) and \
                    not pd.isna(track[Column.ARTIST_YEAR.value]):
                updated_tracks.append(track)
                print(f'Processed track #{i}: it already contains params')
                continue

            artist_name = track[Column.ARTIST_NAME.value].lower()
            cached_res = cls.cache.get(artist_name, None)
            if cached_res is None:
                updated_tracks.append(track)
                continue

            country, sex, year = cached_res[Column.ARTIST_COUNTRY.value], cached_res[Column.ARTIST_SEX.value], \
                                 cached_res[Column.ARTIST_YEAR.value]
            if track[Column.ARTIST_COUNTRY.value] is None:
                track[Column.ARTIST_COUNTRY.value] = country
            if track[Column.ARTIST_SEX.value] is None:
                track[Column.ARTIST_SEX.value] = sex
            if track[Column.ARTIST_YEAR.value] is None:
                track[Column.ARTIST_YEAR.value] = year

            print(f'Processed track #{i} ({track["track_title"]}): new params: {country}, {sex}, {year}')
            updated_tracks.append(track)
            cls._save_cache()

        return pd.DataFrame(updated_tracks)

    @classmethod
    def update_artists_info(cls, tracks: pd.DataFrame) -> pd.DataFrame:
        cls._load_cache()
        updated_tracks = []
        for i, track in tracks.iterrows():
            if not pd.isna(track[Column.ARTIST_COUNTRY.value]) and \
                    not pd.isna(track[Column.ARTIST_SEX.value]) and \
                    not pd.isna(track[Column.ARTIST_YEAR.value]):
                updated_tracks.append(track)
                print(f'Processed track #{i}: it already contains params')
                continue

            country, sex, year = cls.get_artist_info(track[Column.ARTIST_NAME.value])
            if track[Column.ARTIST_COUNTRY.value] is None:
                track[Column.ARTIST_COUNTRY.value] = country
            if track[Column.ARTIST_SEX.value] is None:
                track[Column.ARTIST_SEX.value] = sex
            if track[Column.ARTIST_YEAR.value] is None:
                track[Column.ARTIST_YEAR.value] = year

            print(f'Processed track #{i} ({track["track_title"]}): new params: {country}, {sex}, {year}')
            updated_tracks.append(track)
            cls._save_cache()

        return pd.DataFrame(updated_tracks)

    @classmethod
    def get_artist_info(cls, artist_name: str) -> tuple[Union[str, None], Union[str, None], Union[str, None]]:
        artist_name = artist_name.lower()

        country, sex, year, requests_cnt = None, None, None, 0

        if artist_name in cls.cache:
            info = cls.cache[artist_name]
            country = info[Column.ARTIST_COUNTRY.value]
            sex = info[Column.ARTIST_SEX.value]
            year = info[Column.ARTIST_YEAR.value]
            requests_cnt = info[CACHE_COLUMN_REQUESTS_CNT]

        if country is not None and sex is not None and year is not None:
            return country, sex, year

        if requests_cnt > REQUESTS_LIMIT:
            return country, sex, year

        country_new, sex_new, year_new = cls.try_musicovery(artist_name)
        if country is None and country_new is not None:
            country = country_new
        if sex is None and sex_new is not None:
            sex = sex_new
        if year is None and year_new is not None:
            year = year_new

        if country is None:
            country = cls.try_country_jamendo(artist_name)
        if country is None:
            country = cls.try_country_music_story(artist_name)

        cls.cache[artist_name] = {
            Column.ARTIST_NAME.value: artist_name,
            Column.ARTIST_COUNTRY.value: country,
            Column.ARTIST_SEX.value: sex,
            Column.ARTIST_YEAR.value: year,
            CACHE_COLUMN_REQUESTS_CNT: requests_cnt + 1
        }

        return country, sex, year

    @classmethod
    # https://musicovery.com/api/V5/doc/documentation.php
    def try_musicovery(cls, artist_name: str):

        country, sex, year = None, None, None
        if cls.musicovery_limit_reached:
            return country, sex, year

        sleep(1)

        opener = urllib.request.build_opener()
        tree_search: ET._ElementTree = ET.parse(opener.open(URL_MUSICOVERY_SEARCH % quote_plus(artist_name)))

        if not tree_search or tree_search.getroot() is None:
            return country, sex, year

        has_error = False
        for element in tree_search.getroot().iter(FIELD_MUSICOVERY_CODE, FIELD_MUSICOVERY_ANSWER):
            if element.tag == FIELD_MUSICOVERY_CODE:
                if element.text != '100':
                    print(f'Error response from Musicovery, code: {element.text}')
                    has_error = True

                if element.text == '103':
                    cls.musicovery_limit_reached = True
                    print('Musicovery: too many requests')

            if element.tag == FIELD_MUSICOVERY_ANSWER:
                if element.value != 'valid':
                    print(f'Error response from Musicovery, answer: {element.text}')
                    has_error = True
        if has_error:
            print(f'Error reseponse: {ET.tostring(tree_search.getroot())}')
            return country, sex, year

        artist_id = None
        for element in tree_search.getroot().iter(FIELD_MUSICOVERY_COUNTRY, FIELD_MUSICOVERY_MBID):
            if element.tag == FIELD_MUSICOVERY_COUNTRY:
                country = element.text
            if element.tag == FIELD_MUSICOVERY_MBID:
                artist_id = element.text

        if artist_id is None:
            return country, sex, year

        tree_get_info: ET._ElementTree = ET.parse(opener.open(URL_MUSICOVERY_GET_INFO % artist_id))

        if not tree_get_info or tree_get_info.getroot() is None:
            return country, sex, year

        for element in tree_get_info.getroot().iter(FIELD_MUSICOVERY_COUNTRY, FIELD_MUSICOVERY_SEXE,
                                                    FIELD_MUSICOVERY_YEAR_DEBUT):
            if element.tag == FIELD_MUSICOVERY_COUNTRY:
                country = element.text
            if element.tag == FIELD_MUSICOVERY_SEXE:
                sex = element.text
            if element.tag == FIELD_MUSICOVERY_YEAR_DEBUT:
                year = element.text

        return country, sex, year

    @staticmethod
    # https://developer.jamendo.com/v3.0/artists/locations
    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Officially_assigned_code_elements
    def try_country_jamendo(artist_name: str) -> Union[str, None]:
        r = requests.get(URL_JAMENDO_SEARCH % quote_plus(artist_name)).json()
        if r['headers'] is None or r['headers']['status'] != 'success':
            return None
        if r[FIELD_JAMENDO_RESULTS] is None or len(r[FIELD_JAMENDO_RESULTS]) == 0:
            return None
        artist = r[FIELD_JAMENDO_RESULTS][0]
        if artist[FIELD_JAMENDO_LOCATIONS] is None or len(artist[FIELD_JAMENDO_LOCATIONS]) == 0:
            return None
        return artist[FIELD_JAMENDO_LOCATIONS][0][FIELD_JAMENDO_COUNTRY]

    @staticmethod
    # http://developers.music-story.com/developers/artist#c_artists
    def try_country_music_story(artist_name: str) -> Union[str, None]:
        # TODO: разобраться
        r = requests.get('http://api.music-story.com/en/artist/114832')
        return None

    @classmethod
    def _load_cache(cls) -> dict[str, dict[str, Any]]:
        if os.path.isfile(FILE_ARTISTS_INFO):
            df = pd.read_csv(FILE_ARTISTS_INFO)
            artists_info = {}
            for _, row in df.iterrows():
                artists_info[row[Column.ARTIST_NAME.value]] = row.to_dict()
            cls.cache = artists_info
            return cls.cache

        df: pd.DataFrame = pd.DataFrame(data=[], columns=CACHE_COLUMNS)
        ensure_dirs_exist([Path(FILE_ARTISTS_INFO).parent])
        df.to_csv(FILE_ARTISTS_INFO, index=False)
        return {}

    @classmethod
    def _save_cache(cls):
        pd.DataFrame(data=cls.cache.values(), columns=CACHE_COLUMNS).to_csv(FILE_ARTISTS_INFO, index=False)

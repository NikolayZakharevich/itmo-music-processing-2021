import os
import random
import re

import pandas as pd

import assignments_analog
import assignments_original
import assignments_random

DIR_BASE = os.getcwd() + os.pathsep
DIR_DATA = '/Users/n.zakharevich/image-generation/cpanel/dataset/data'

FILE_TRACKS = os.path.join(DIR_DATA, 'tracks.csv')
DIR_IMAGES = os.path.join(DIR_DATA, 'images')

COLUMN_GENRE = 'ym_genre'
COLUMN_TRACK_ID = 'ym_track_id'
COLUMN_COVER_URL = 'cover_url'

URL_DATA_PREFIX = 'https://sky4uk.xyz/static/data/'

URL_AUDIO = URL_DATA_PREFIX + 'audios/%d.mp3'  # track_id
URL_COVER_GENERATED = URL_DATA_PREFIX + 'images/%d_gen.jpg'  # track_id
URL_COVER_GENERATED_OTHER = URL_DATA_PREFIX + 'images-cover-art-generation/data_2/image-%d.jpg'  # track__id


def load():
    tracks_df = pd.read_csv(FILE_TRACKS).set_index([COLUMN_TRACK_ID])

    covers = tracks_df.to_dict('dict')[COLUMN_COVER_URL]
    genres = tracks_df.to_dict('dict')[COLUMN_GENRE]

    result = []
    for filename_image in os.listdir(DIR_IMAGES):
        matches = re.search(r'(\d+)_gen.jpg', filename_image)
        if matches is None:
            continue
        track_id = int(matches.group(1))
        if track_id is None:
            continue
        if track_id not in covers or track_id not in genres:
            print(f'Missing row for track_id: {track_id}')
            continue

        genre = genres[track_id]
        url_audio = URL_AUDIO % track_id
        url_cover = 'https://' + covers[track_id]
        url_cover_generated = URL_COVER_GENERATED % track_id
        result.append({
            'track_id': track_id,
            'genre': genre,
            'url_audio': url_audio,
            'url_cover': url_cover,
            'url_cover_generated': url_cover_generated
        })

    return result


def my_vs_original():
    data = load()

    result = []
    for item in data:
        result.append({
            'audio': item['url_audio'],
            'cover_original': item['url_cover'],
            'cover_generated': item['url_cover_generated'],
        })
    pd.DataFrame(result).to_csv('my_vs_original.csv', index=False)


def my_vs_album_cover_generation():
    data = load()

    result = []

    for item in data:
        result.append({
            'audio': item['url_audio'],
            'cover_generated_my': item['url_cover_generated'],
            'cover_generated_another': URL_COVER_GENERATED_OTHER % random.randint(1, 1000)
        })
    pd.DataFrame(result).to_csv('my_vs_another.csv', index=False)


def run_all():
    assignments_analog.run()
    assignments_original.run()
    assignments_random.run()


def get_results_all():
    assignments_analog.get_results()
    assignments_original.get_results()
    assignments_random.get_results()


if __name__ == '__main__':
    # my_vs_original()
    # my_vs_album_cover_generation()
    # run_all()
    get_results_all()

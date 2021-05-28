# import launch
import os
from google_trans_new import google_translator
import pandas as pd

from pathlib import Path

from app.columns import Column
from app.utils import ensure_dirs_exist
from lyrics_extractor import SongLyrics
from config import DIR_DATA_LYRICS, to_absolute_path, GCS_API_KEY, GCS_ENGINE_ID

DIR_LYRICS = 'lyrics/'
FILE_KEYWORDS = 'keywords.csv'
N_KEYWORDS = 10


def update_lyrics():
    lyrics_extractor = SongLyrics(GCS_API_KEY, GCS_ENGINE_ID)

    lyrics_dir = Path(to_absolute_path(DIR_DATA_LYRICS))
    ensure_dirs_exist([lyrics_dir])

    tracks = pd.read_csv('data/tracks.csv')
    data = []
    for _, track in tracks.iterrows():
        if pd.isna(track.lyrics_path) or not os.path.isfile(track.lyrics_path):
            data = lyrics_extractor.get_lyrics(track[Column.TRACK_TITLE])
            if data and data['lyrics']:
                lyrics_filename = f'{track.id}.txt'
                with open(lyrics_dir / Path(lyrics_filename), "w") as lyrics_file:
                    print(data['lyrics'], file=lyrics_file)
                    track.lyrics_path = DIR_DATA_LYRICS + lyrics_filename
            else:
                tracks.lyrics_path = None
        else:
            print('Lyrics file already exists')

        data.append(track)

    tracks = pd.DataFrame(data=data, columns=tracks.columns)
    print(tracks)


def load_local_embedding_distributor():
    # see https://github.com/swisscom/ai-research-keyphrase-extraction
    # replace with launch.load_local_embedding_distributor()
    pass


def load_local_corenlp_pos_tagger():
    # see https://github.com/swisscom/ai-research-keyphrase-extraction
    # replace with launch.load_local_embedding_distributor()
    pass


def extract_keyphrases(embedding_distrib, ptagger, raw_text, N, lang, beta=0.55, alias_threshold=0.7):
    # see https://github.com/swisscom/ai-research-keyphrase-extraction
    # replace with launch.extract_keyphrases()

    """
    Method that extract a set of keyphrases

    :param embedding_distrib: An Embedding Distributor object see @EmbeddingDistributor
    :param ptagger: A Pos Tagger object see @PosTagger
    :param raw_text: A string containing the raw text to extract
    :param N: The number of keyphrases to extract
    :param lang: The language
    :param beta: beta factor for MMR (tradeoff informativness/diversity)
    :param alias_threshold: threshold to group candidates as aliases
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    pass


def update_keywords(keywords_df: pd.DataFrame, dir_lyrics: str):
    embedding_distributor = load_local_embedding_distributor()
    pos_tagger = load_local_corenlp_pos_tagger()

    translator = google_translator()

    updated_keywords = keywords_df.to_dict('list')
    track_ids = set(updated_keywords['track_id'])
    filenames = os.listdir(dir_lyrics)
    for i, filename in enumerate(filenames):
        track_id, _ = os.path.splitext(filename)
        print(f'Processing track {i + 1}/{len(filenames)}. Id: {track_id}...')

        if int(track_id) in track_ids:
            print(f'Already have keywords for track {track_id}')
            continue

        lyrics_path = DIR_LYRICS + filename
        with open(lyrics_path) as f:
            try:
                lyrics = f.read().replace('\n', ' ').replace(' ', ' ')
                if not lyrics:
                    continue

                lyrics_en = translator.translate(lyrics)

                if not lyrics_en:
                    continue

                keywords, weights, _ = extract_keyphrases(
                    embedding_distributor, pos_tagger, lyrics_en, N_KEYWORDS, 'en')

                keywords_str = ";".join(keywords)
                weights_str = ";".join(map(str, weights))
                updated_keywords['track_id'].append(track_id)
                updated_keywords['keywords'].append(keywords_str)
                updated_keywords['weights'].append(weights_str)

            except Exception as e:
                print(track_id, e)
                pass

    return pd.DataFrame(
        data=updated_keywords,
        columns=keywords_df.columns
    )

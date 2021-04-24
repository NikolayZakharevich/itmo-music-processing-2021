import os
import re
from pathlib import Path
from pickle import dump, load
from typing import List

import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.preprocessing import LabelBinarizer

from app.common.columns import Column, EMOTIONS
from app.features.features import extract_audio_features_v1
from operator import itemgetter

SECOND_MILLIS = 1000


def remove_album_from_name(audio_paths):
    for path in audio_paths:
        credentials_search = re.search(fr'/(\d+):(\d+).mp3', path)
        if not credentials_search:
            continue
        album_id = credentials_search.group(1)

        if os.path.isfile(path):
            new_path = path.replace(f'{album_id}:', '')
            print(path, new_path)
            os.rename(path, new_path)


def remove_all_audios_but(dir_audios: str, audio_paths_to_keep: pd.Series):
    dir_audios = add_trailing_slash(dir_audios)
    paths_to_keep = set(audio_paths_to_keep.tolist())

    audio_paths_all = os.listdir(dir_audios)
    for i, filename in enumerate(audio_paths_all):
        audio_path = dir_audios + filename
        print(f'Processing file {audio_path} ({i + 1} / {len(audio_paths_all)})')

        if audio_path in paths_to_keep:
            continue
        elif os.path.isfile(audio_path):
            os.remove(audio_path)


def check_audio_files_exist(tracks: pd.DataFrame) -> List[str]:
    missing_paths = []

    for _, track in tracks.iterrows():
        track_id = track[Column.YM_TRACK_ID.value]
        audio_path = os.getcwd() + '/' + track.audio_path
        if not os.path.isfile(audio_path):
            print(f'Missing audio file for track {track_id}: expected audio path: {audio_path}')
            missing_paths.append(audio_path)
    return missing_paths


# Returns path to copied audio
def copy_first_30_seconds(audio_path, dest_path=None) -> str:
    N_SECONDS = 30

    name, ext = audio_path.split('/')[-1].split('.')
    sound = AudioSegment.from_file(audio_path, ext)

    # len() and slicing are in milliseconds
    length_ms = len(sound)
    high = min(N_SECONDS * SECOND_MILLIS, length_ms)
    part = sound[:high]

    if dest_path is None:
        dest_path = audio_path.replace(name, f'{name}_0-{high / SECOND_MILLIS}')

    part.export(dest_path, format(ext))
    return dest_path


def filter_emotion_train_v1(tracks: pd.DataFrame) -> pd.DataFrame:
    tracks = tracks.dropna(subset=['emotions'])
    tracks = tracks[tracks.apply(lambda t: '|' not in t.emotions, axis=1)]
    return tracks.groupby(['emotions']).head(200)


def get_short_audio_path(audio_path: str):
    return audio_path.replace('audios', 'audios_30s')


def dump_data(track_ids: np.array, x: np.array, y: np.array, output_path: str):
    data = {'track_ids': track_ids, 'x': x, 'y': y}
    with open(output_path, 'wb') as f:
        dump(data, f)
    return data


def save_extracted_features_batch(
        tracks_batch: pd.DataFrame,
        features_shape,
        output_path: str,
        lb: LabelBinarizer,
        use_short_audio: bool = True
):
    tracks_count = len(tracks_batch)
    X = np.zeros((tracks_count,) + features_shape, dtype=np.float32)
    y = lb.transform(tracks_batch[Column.EMOTIONS.value])
    track_ids = tracks_batch[Column.YM_TRACK_ID.value].to_numpy(dtype=np.int32)

    idx = 0
    for _, track in tracks_batch.iterrows():
        track_id = track[Column.YM_TRACK_ID.value]
        audio_path = track[Column.AUDIO_PATH.value]

        print(f'Track #{idx + 1} out of {len(tracks_batch)}. Track id: {track_id}')
        audio_path = get_short_audio_path(audio_path) if use_short_audio else audio_path
        if not os.path.isfile(audio_path):
            print(f'Track #{idx + 1}: audio file {audio_path} is missing')
            continue
        X[idx] = extract_audio_features_v1(audio_path, features_shape)
        idx += 1

    print(track_ids, y)
    dump_data(track_ids, X, y, output_path)


def save_extracted_features(
        tracks: pd.DataFrame,
        dump_dir: str,
        use_short_audio: bool = True,
        n_batches: int = 10
):
    dump_dir_path = Path(dump_dir)
    dump_dir_path.mkdir(parents=True, exist_ok=True)

    lb = LabelBinarizer()
    lb.fit(EMOTIONS)

    # determining features_shape
    first_track = tracks.iloc[0]
    first_track_audio_path = get_short_audio_path(first_track.audio_path) if use_short_audio else first_track.audio_path
    first_track_x = extract_audio_features_v1(first_track_audio_path)
    features_shape = first_track_x.shape

    for i, tracks_batch in enumerate(np.array_split(tracks, n_batches)):
        print(f'Processing batch #{i + 1}...')
        batch_output_path = dump_dir_path / Path(f'batch-{i}.pkl')

        save_extracted_features_batch(
            tracks_batch=tracks_batch,
            features_shape=features_shape,
            output_path=str(batch_output_path),
            lb=lb,
            use_short_audio=use_short_audio
        )


def load_data_from_dumps(tracks: pd.DataFrame, dump_dir: str):
    tracks_data = {}

    dump_dir = add_trailing_slash(dump_dir)
    dump_files = list(map(lambda filename: dump_dir + filename, os.listdir(dump_dir)))
    dump_files_iter = iter(dump_files)

    for _, track in tracks.iterrows():
        while track[Column.YM_TRACK_ID.value] not in tracks_data and \
                (dump_file := next(dump_files_iter, None)) is not None:
            with open(dump_file, 'rb') as f:
                track_ids, x, y = itemgetter('track_ids', 'x', 'y')(load(f))
                for i in range(len(track_ids)):
                    tracks_data[track_ids[i]] = x[i], y[i]
    return tracks_data


def compose(f, g):
    return lambda x: f(g(x))


def add_trailing_slash(path: str) -> str:
    return os.path.join(path, '')

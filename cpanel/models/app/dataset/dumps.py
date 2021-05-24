import io
import os
import pickle as pkl
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request

import numpy as np

from app.common.columns import DumpColumn
from app.common.tracks import get_audio_url, get_audio_path_default
from app.common.utils import add_trailing_path_separator
from app.features.features import extract_audio_features_v1_chunks
from config import DIR_FEATURES_V1, DIR_FEATURES_V1_SINGLE, DUMP_FILE_EXTENSION_BATCH, DUMP_FILE_EXTENSION_SINGLE

N_BATCHES = 20

BATCH_NAME_PREFIX = 'batch-'

STORAGE_REQUEST_HEADERS = {'User-Agent': 'Mozilla/5.0'}

ERROR_MESSAGE_BATCH_LOAD_EXCEPTION = 'Batch load exception: '
ERROR_MESSAGE_INVALID_DUMP_DIR = 'Invalid dump dir: '


def save_to_separate_files(track_ids: set[int], to_dump_dir=DIR_FEATURES_V1_SINGLE):
    """
    :param track_ids:
    :param to_dump_dir:
    :return:
    """
    to_dump_dir = add_trailing_path_separator(to_dump_dir)

    requested_tracks_count = len(track_ids)
    print(f'Requested tracks number: {requested_tracks_count}')

    tracks_ids_by_remainder = group_ids_by_remainder(track_ids)

    for batch_idx in range(N_BATCHES):
        if len(tracks_ids_by_remainder[batch_idx]) == 0:
            continue

        batch_tracks_features = load_from_dumps(batch_idx, DIR_FEATURES_V1)
        print_batch_info(batch_idx, len(batch_tracks_features), len(tracks_ids_by_remainder[batch_idx]))

        for track_id in tracks_ids_by_remainder[batch_idx]:
            if track_id not in batch_tracks_features:
                try:
                    features = extract_features(track_id)
                except Exception as e:
                    print(e)
                    continue
            else:
                features = batch_tracks_features[track_id]
            if len(features) > 0:
                save_single_track_features(track_id, features, to_dump_dir)

        for track_id in tracks_ids_by_remainder[batch_idx]:
            audio_path = get_audio_path_default(track_id)
            if os.path.exists(audio_path):
                os.remove(audio_path)


def get_tracks_features_v1(track_ids_requested: set[int], dump_dir=DIR_FEATURES_V1) -> dict[int, list[np.ndarray]]:
    requested_tracks_count = len(track_ids_requested)
    print(f'Requested tracks number: {requested_tracks_count}')

    tracks_features: dict[int, list[np.ndarray]] = load_from_dumps_or_extract(track_ids_requested, dump_dir)
    received_tracks_count = len(tracks_features)
    print(f'Received tracks number: {received_tracks_count}')

    track_ids_left = track_ids_requested - set(tracks_features.keys())
    if track_ids_left:
        print('Download audios for tracks:')
        for track_id in track_ids_left:
            print(track_id)

    return tracks_features


def load_from_dumps_or_extract(track_ids_requested: set[int], dump_dir=DIR_FEATURES_V1) -> dict[int, list[np.ndarray]]:
    """
    Loads features from dump batches or extracting them from audio
    :param track_ids_requested:
    :param dump_dir:
    :return:
    """
    tracks_features: dict[int, list[np.ndarray]] = {}
    dump_dir = add_trailing_path_separator(dump_dir)
    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)

    tracks_ids_by_remainder = group_ids_by_remainder(track_ids_requested)

    for batch_idx in range(N_BATCHES):
        if len(tracks_ids_by_remainder[batch_idx]) == 0:
            continue

        batch_tracks_features = load_from_dumps(batch_idx, dump_dir)
        print_batch_info(batch_idx, len(batch_tracks_features), len(tracks_ids_by_remainder[batch_idx]))

        batch_track_ids = list(batch_tracks_features.keys())
        batch_features = list(batch_tracks_features.values())

        batch_changed: bool = False
        for track_id in tracks_ids_by_remainder[batch_idx]:
            if track_id not in batch_tracks_features:
                try:
                    features = extract_features(track_id)
                    tracks_features[track_id] = features
                    batch_track_ids.append(track_id)
                    batch_features.append(features)
                    batch_changed = True
                except Exception as e:
                    print(e)
                    continue
            else:
                tracks_features[track_id] = batch_tracks_features[track_id]

        if batch_changed:
            save_batch(batch_idx, batch_track_ids, batch_features)

        for track_id in tracks_ids_by_remainder[batch_idx]:
            audio_path = get_audio_path_default(track_id)
            if os.path.exists(audio_path):
                os.remove(audio_path)

    return tracks_features


def load_from_dumps(batch_idx: int, dump_dir: str) -> dict[int, list[np.ndarray]]:
    batch_path: str = get_batch_path(batch_idx, dump_dir)
    batch_track_ids, batch_features = [], []

    if os.path.isfile(batch_path):
        try:
            loaded_track_ids, loaded_features = load_track_ids_and_features_from_dump(batch_path)
            if loaded_track_ids is not None:
                batch_track_ids = loaded_track_ids
            else:
                print(get_missing_dump_column_message(batch_path, 'dump_track_ids'))
            if loaded_features is not None:
                batch_features = loaded_features
            else:
                print(get_missing_dump_column_message(batch_path, 'dump_features'))
        except Exception as e:
            print(ERROR_MESSAGE_BATCH_LOAD_EXCEPTION, e)

    batch_tracks_features: dict[int, list[np.ndarray]] = dict(zip(batch_track_ids, batch_features))
    return batch_tracks_features


def load_track_ids_and_features_from_dump(batch_file: str) -> tuple[
    Optional[list[int]], Optional[list[list[np.ndarray]]]]:
    """
    Open end parse batch file
    :param batch_file:
    :return:
    """
    with open(batch_file, 'rb') as f:
        dump_dict: dict = pkl.load(f)
        dump_track_ids = dump_dict.get(DumpColumn.TRACK_IDS.value, None)
        dump_features = dump_dict.get(DumpColumn.FEATURES.value, None)
        return dump_track_ids, dump_features


def extract_features(track_id: int) -> list[np.ndarray]:
    """
    Extracting features for track with id=track_id
    :param track_id:
    :return:
    """
    audio_url: str = get_audio_url(track_id)
    audio_path: Path = Path(get_audio_path_default(track_id))

    # If theres is no audio with such id, download it from storage server
    if not audio_path.exists():
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        r = Request(audio_url, headers=STORAGE_REQUEST_HEADERS)
        audio_buffer = io.BytesIO(urlopen(r).read())
        audio_path.write_bytes(audio_buffer.getbuffer())

    return extract_audio_features_v1_chunks(audio_path)


def save_batch(batch_idx: int, dump_track_ids: list[int], dump_features: list[list[np.ndarray]],
               dump_dir=DIR_FEATURES_V1):
    """
    Dumps features to batch file
    :param batch_idx:      index of batch to save
    :param dump_track_ids: tracks' ids
    :param dump_features:  tracks' features
    :param dump_dir:       directory to save dump
    :return:
    """
    dump_dir = add_trailing_path_separator(dump_dir)
    dump_dict = {
        DumpColumn.TRACK_IDS.value: dump_track_ids,
        DumpColumn.FEATURES.value: dump_features
    }

    if os.path.isfile(dump_dir):
        print(ERROR_MESSAGE_INVALID_DUMP_DIR)
        return

    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)

    output_path = get_batch_path(batch_idx, dump_dir)
    with open(output_path, 'wb') as f:
        pkl.dump(dump_dict, f)


def save_single_track_features(track_id: int, features: list[np.ndarray], dump_dir=DIR_FEATURES_V1_SINGLE):
    """
    :param track_id: track id to save
    :param features: track's features
    :param dump_dir: directory to save dump
    :return:
    """
    dump_dir = add_trailing_path_separator(dump_dir)

    if os.path.isfile(dump_dir):
        print(ERROR_MESSAGE_INVALID_DUMP_DIR)
        return

    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)

    output_path = get_single_track_features_path(track_id, dump_dir)
    np.save(output_path, features)


def get_batch_path(batch_idx: int, dump_path: str = DIR_FEATURES_V1) -> str:
    """
    :param batch_idx:
    :param dump_path:
    :return: batch file name
    """
    return f'{dump_path}{BATCH_NAME_PREFIX}{str(batch_idx)}.{DUMP_FILE_EXTENSION_BATCH}'


def get_single_track_features_path(track_id: int, dump_dir=DIR_FEATURES_V1_SINGLE) -> str:
    """
    :param track_id:
    :param dump_dir:
    :return: single track features file name
    """
    return f'{dump_dir}{track_id}.{DUMP_FILE_EXTENSION_SINGLE}'


def get_missing_dump_column_message(dump_file: str, missing_attribute: str) -> str:
    """
    Message printed if dump file is invalid format
    :param dump_file:         name of opened dump file
    :param missing_attribute: attribute which is missing in dict
    :return:
    """
    return f'Dump file {dump_file} is invalid: column {missing_attribute} is missing'


def dump_file_exists(track_id: int, dump_dir=DIR_FEATURES_V1_SINGLE) -> bool:
    """
    Determines if there is a dump file for track with such id
    :param track_id:
    :param dump_dir:
    :return:
    """
    return os.path.isfile(get_single_track_features_path(track_id, dump_dir))


def group_ids_by_remainder(track_ids: set[int], n_batches=N_BATCHES) -> list[set[int]]:
    tracks_ids_by_remainder = [set() for _ in range(n_batches)]
    for track_id in track_ids:
        rem = track_id % N_BATCHES
        tracks_ids_by_remainder[rem].add(track_id)
    return tracks_ids_by_remainder


def print_batch_info(batch_idx: int, n_tracks_in_batch: int, n_tracks_requested_from_batch: int):
    print(f'Number of tracks in batch-{batch_idx}: {n_tracks_in_batch}')
    print(f'Number of tracks requested from batch-{batch_idx}: {n_tracks_requested_from_batch}')

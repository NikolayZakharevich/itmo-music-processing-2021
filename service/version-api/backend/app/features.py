import os
import random
import string
import warnings
from typing import List, Union, Optional

import librosa as lbr
import numpy as np

WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

SECOND_IN_MILLIES = 1000
MINUTE_LENGTH = 1292

DUMP_FILENAME_LENGTH = 16
DUMP_FILENAME_EXTENSION = '.npy'


def get_dump_path(filename_without_extension: str, dump_dir: str) -> str:
    """
    :param filename_without_extension:
    :param dump_dir:
    :return:
    >>> get_dump_path('asabsdfgasg', '/var/tmp')
        /var/tmp/asabsdfgasg.npy
    """
    return os.path.join(dump_dir, filename_without_extension + DUMP_FILENAME_EXTENSION)


def extract_audio_features(filename: Union[str, bytes, os.PathLike]) -> List[np.ndarray]:
    """
    :param filename:
    :return: list of features for each full minute
    """
    print(f'Extracting audio features from file {filename}...')
    with warnings.catch_warnings():
        new_input, sample_rate = lbr.load(filename, mono=True)
        features: np.ndarray = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T
    features[features == 0] = 1e-6
    return split_equal_chunks(np.log(features), MINUTE_LENGTH)


def save_audio_features(features: List[np.ndarray], dump_dir: str) -> Optional[str]:
    """
    :param track_id: track id to save
    :param features: track's features
    :param dump_dir: directory to save dump
    :return: dump filename without extension
    """
    if os.path.isfile(dump_dir):
        return None

    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)

    filename_without_extension = ''.join(
        random.choice(string.ascii_letters) for _ in range(DUMP_FILENAME_LENGTH))
    output_path = get_dump_path(filename_without_extension, dump_dir)
    np.save(output_path, features)
    return filename_without_extension


def split_equal_chunks(l: List, chunk_size: int):
    """
    Ignores tail after last chunk
    >>> split_equal_chunks(list(range(10)), 3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> split_equal_chunks(list(range(10)), 2)
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    :param l:
    :param chunk_size:
    :return:
    """
    return [l[i - chunk_size: i] for i in range(chunk_size, len(l) + 1, chunk_size)]

import os
from typing import Optional, Callable, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from app.dataset.dumps import get_single_track_features_path, dump_file_exists
from app.common.tracks import get_audio_path_default, audio_file_exists
from app.common.utils import compose3
from app.features.features import N_MELS, MINUTE_LENGTH, \
    extract_audio_features_v1_chunks
from config import DIR_FEATURES_V1_SINGLE

BATCH_SIZE = 9
TEST_SPLIT_SEED = 2021


def get_dataloaders_split(
        track_ids: list[int],
        labels: dict[int, str],
        label_names: Optional[list[str]] = None,
        batch_size: int = BATCH_SIZE,
        test_split_size: float = 0.1,
        test_split_seed: int = TEST_SPLIT_SEED

) -> tuple[DataLoader, DataLoader]:
    if label_names is None:
        label_names = list(set(labels.values()))

    track_ids_train, track_ids_val = train_test_split(track_ids, test_size=test_split_size, shuffle=True,
                                                      random_state=test_split_seed)

    dataloaders = []
    for track_ids_split in [track_ids_train, track_ids_val]:
        labels_split = {track_id: labels[track_id] for track_id in track_ids_split}
        dataset = TracksDataset(
            track_ids=track_ids_split,
            labels=labels_split,
            transform=compose3(
                EncodeLabel(label_names),
                MelSpectrogramCached(),
                ExtractFirstMinute()
            )
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        dataloaders.append(dataloader)
    return dataloaders[0], dataloaders[1]


class EncodeLabel(object):
    le: LabelEncoder

    def __init__(self, label_names: list[str]):
        self.le = LabelEncoder()
        self.le.fit(label_names)

    def __call__(self, sample: tuple[int, str]) -> tuple[int, int]:
        track_id, label = sample
        return track_id, self.le.transform([label])[0]


class ExtractFirstMinute(object):
    def __init__(self):
        pass

    def __call__(self, sample: tuple[Tensor, int]) -> tuple[Tensor, int]:
        features, label = sample
        return features[0], label


class MelSpectrogramCached(object):
    dump_dir: Union[str, bytes, os.PathLike]
    chunk_size: int
    n_mels: int

    def __init__(
            self,
            dump_dir: Union[str, bytes, os.PathLike] = DIR_FEATURES_V1_SINGLE,
            chunk_size: int = MINUTE_LENGTH,
            n_mels: int = N_MELS
    ):
        self.dump_dir = dump_dir
        self.chunk_size = chunk_size
        self.n_mels = n_mels

    def __call__(self, sample: tuple[int, int]) -> tuple[Tensor, int]:
        track_id, label = sample
        if dump_file_exists(track_id, self.dump_dir):
            return torch.tensor(np.load(get_single_track_features_path(track_id))), label

        if not audio_file_exists(track_id):
            print(f'Missing audio file for track with id = {track_id}')
            return torch.zeros((1, self.chunk_size, self.n_mels)), label

        features = extract_audio_features_v1_chunks(get_audio_path_default(track_id))
        return torch.tensor(features), label


class TracksDataset(Dataset):
    track_ids: list[int]
    labels: dict[int, str]

    def __init__(
            self,
            track_ids: list[int],
            labels: dict[int, str],
            transform: Optional[Callable] = None
    ):
        self.track_ids = track_ids
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx) -> tuple[int, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        track_id = self.track_ids[idx]
        sample = track_id, self.labels[track_id]
        if self.transform:
            sample = self.transform(sample)

        return sample

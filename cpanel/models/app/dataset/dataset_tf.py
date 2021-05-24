import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.utils.data_utils import Sequence

from app.dataset.dumps import get_single_track_features_path

from abc import ABCMeta, abstractmethod

BATCH_SIZE_DEFAULT = 32
DIM_DEFAULT = (1292, 128)


class AbstractTracksGenerator(Sequence):
    __metaclass__ = ABCMeta

    track_ids: list[int]
    labels: dict

    batch_size: int
    dim: tuple
    shuffle: bool

    mlb: MultiLabelBinarizer

    indexes: np.ndarray

    def __init__(
            self,
            track_ids,
            labels: dict,
            label_names: list,
            batch_size: int = BATCH_SIZE_DEFAULT,
            dim: tuple = DIM_DEFAULT,
            shuffle: bool = True
    ):
        mlb = MultiLabelBinarizer()
        mlb.fit([label_names])

        self.track_ids = track_ids
        self.labels = labels

        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.mlb = mlb

        self._on_init_end()
        self.on_epoch_end()

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        return int(np.floor(len(self._items()) / self.batch_size))

    def __getitem__(self, index):
        """
        'Generate one batch of data
        :param index:
        :return:
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self._data_generation([self._items()[i] for i in indexes])

    def on_epoch_end(self):
        """
        Method called at the end of every epoch
        :return:
        """
        self.indexes = np.arange(len(self._items()))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @abstractmethod
    def _data_generation(self, batch_items: list) -> tuple[np.ndarray, np.ndarray]:
        """
        X.shape: (n_samples, *dim)
        y.shape: (n_samples, len(label_names))
        :param batch_items:
        :return: X, y
        """
        pass

    def _on_init_end(self):
        pass

    def _items(self) -> list:
        return self.track_ids


class TracksGenerator(AbstractTracksGenerator):
    track_ids_with_minute_indexes: list[tuple[int, int]]  # (track ID, minute index)

    def _on_init_end(self):
        self.track_ids_with_minute_indexes = []
        for track_id in self.track_ids:
            for minute_index in range(len(np.load(get_single_track_features_path(track_id)))):
                self.track_ids_with_minute_indexes.append((track_id, minute_index))

    def _items(self) -> list[tuple[int, int]]:
        return self.track_ids_with_minute_indexes

    def _data_generation(self, batch_items: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """
        X.shape: (n_samples, *dim)
        y.shape: (n_samples, len(label_names))
        :param batch_items:
        :return: X, y
        """

        X_list = []
        y_labels = []

        for (track_id, minute_index) in batch_items:
            features = np.load(get_single_track_features_path(track_id))
            X_list.append(features[minute_index])
            y_labels.append(self.labels[track_id])

        return np.array(X_list), self.mlb.transform(y_labels)


class TracksFirstMinuteGenerator(AbstractTracksGenerator):

    def _on_init_end(self):
        pass

    def _items(self) -> list[int]:
        return self.track_ids

    def _data_generation(self, batch_items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        X.shape: (n_samples, *dim)
        y.shape: (n_samples, len(label_names))
        :param batch_items:
        :return: X, y
        """

        X_list = []
        y_labels = []

        for track_id in batch_items:
            features = np.load(get_single_track_features_path(track_id))
            X_list.append(features[0])
            y_labels.append(self.labels[track_id])

        return np.array(X_list), self.mlb.transform(y_labels)

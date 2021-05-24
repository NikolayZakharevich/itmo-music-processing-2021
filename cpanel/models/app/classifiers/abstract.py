from os import PathLike
from typing import Iterable, Optional, Union, Type

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.utils.data_utils import Sequence

from app.dataset.dataset_tf import AbstractTracksGenerator, TracksGenerator

BATCH_SIZE_DEFAULT = 9
MONITOR_METRIC_DEFAULT = 'val_loss'
TEST_SIZE_DEFAULT = 0.1

TEST_SPLIT_SEED = 42


class AbstractClassifier:
    model: Model
    label_names = []

    # Default constants:
    epochs: int
    batch_size: int = BATCH_SIZE_DEFAULT
    monitor_metric = MONITOR_METRIC_DEFAULT
    test_size: int = TEST_SIZE_DEFAULT

    def __init__(self, model: Model):
        self.model = model

    @classmethod
    def label_indexes_to_names(cls, label_indexes: Iterable[int]) -> list[str]:
        pass

    @classmethod
    def build_model(cls, n_features):
        pass

    @classmethod
    def _fit_impl(
            cls,
            model: Model,
            generator_train: Sequence,
            generator_val: Sequence,
            output_path: Union[str, bytes, PathLike],
            batch_size: int,
            epochs: int,
            monitor_metric: str
    ) -> tuple[Model, History]:
        pass

    @classmethod
    def train_model(
            cls,
            model: Model,
            track_ids: list[int],
            labels: dict,
            output_path: str,
            tracks_generator_class: Type[AbstractTracksGenerator] = TracksGenerator,
            batch_size: Optional[int] = None,
            epochs: Optional[int] = None,
            test_size: Optional[float] = None,
            monitor_metric: Optional[str] = None
    ):
        if batch_size is None:
            batch_size = cls.batch_size
        if epochs is None:
            epochs = cls.epochs
        if monitor_metric is None:
            monitor_metric = cls.monitor_metric
        if test_size is None:
            test_size = cls.test_size

        params = {
            'label_names': cls.label_names,
            'dim': (1292, 128),
            'batch_size': batch_size,
            'shuffle': True
        }

        track_ids_train, track_ids_val = train_test_split(track_ids, test_size=test_size, shuffle=True,
                                                          random_state=TEST_SPLIT_SEED)

        labels_train = {track_id: labels[track_id] for track_id in track_ids_train}
        labels_val = {track_id: labels[track_id] for track_id in track_ids_val}

        generator_train = tracks_generator_class(track_ids_train, labels_train, **params)
        generator_val = tracks_generator_class(track_ids_val, labels_val, **params)

        print('Training...')
        model, history = cls._fit_impl(
            model=model,
            generator_train=generator_train,
            generator_val=generator_val,
            output_path=output_path,
            batch_size=batch_size,
            epochs=epochs,
            monitor_metric=monitor_metric
        )
        return model, history

    def predict_top_1(self, X: np.ndarray) -> list:
        return list(map(lambda x: x[0], self.predict_top_k(X, k=1)))

    def predict_top_k(self, X: np.ndarray, k=3) -> list[list]:
        return list(map(
            lambda p: self.label_indexes_to_names(np.argpartition(p, -k)[-k:]),
            self.model.predict(X, batch_size=self.batch_size))
        )

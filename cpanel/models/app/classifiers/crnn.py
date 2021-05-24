from os import PathLike
from typing import List, Iterable, Union

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation, \
    TimeDistributed, Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import TopKCategoricalAccuracy
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.ops import math_ops

from app.classifiers.abstract import AbstractClassifier

N_LAYERS = 1
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
BATCH_SIZE = 1024
EPOCHS = 100


class CrnnClassifier(AbstractClassifier):
    monitor_metric = 'val_accuracy'

    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE

    n_layers: int = N_LAYERS
    filter_length: int = FILTER_LENGTH
    conv_filter_count: int = CONV_FILTER_COUNT

    @classmethod
    def label_indexes_to_names(cls, label_indexes: Iterable[int]) -> List[str]:
        return list(map(lambda g: cls.label_names[g], label_indexes))

    @classmethod
    def build_model(cls, n_features):
        n_labels = len(cls.label_names)

        print('Building model...')
        input_shape = (None, n_features)
        model_input = Input(input_shape, name='input')
        layer = model_input
        for i in range(N_LAYERS):
            layer = Convolution1D(
                filters=CONV_FILTER_COUNT,
                kernel_size=FILTER_LENGTH,
                name='convolution_' + str(i + 1)
            )(layer)
            layer = BatchNormalization(momentum=0.9)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling1D(2)(layer)
            layer = Dropout(0.5)(layer)

        layer = TimeDistributed(Dense(n_labels))(layer)
        time_distributed_merge_layer = Lambda(
            function=lambda x: K.mean(x, axis=1),
            output_shape=lambda shape: (shape[0],) + shape[2:],
            name='output_merged'
        )
        layer = time_distributed_merge_layer(layer)
        layer = Activation('softmax', name='output_realtime')(layer)
        model_output = layer
        model = Model(model_input, model_output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy', TopKCategoricalAccuracy(3, name='top3-accuracy')]
        )
        model.summary()
        return model

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
    ):
        history = model.fit(
            generator_train,
            validation_data=generator_val,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=1,
            callbacks=[
                ModelCheckpoint(
                    output_path,
                    save_best_only=True,
                    monitor=monitor_metric,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.5,
                    patience=10,
                    min_delta=0.01,
                    verbose=1
                )
            ]
        )
        return model, history

    @classmethod
    def load(cls, model_path: str):
        model = load_model(model_path)
        return cls(model)


def custom_categorical_crossentropy(y_true, y_pred):
    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return -K.sum(y_true * K.log(y_pred), -1)

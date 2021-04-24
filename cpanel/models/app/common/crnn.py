from typing import List, Iterable

import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation, \
    TimeDistributed, Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from app.common.view import show_history
from app.features.features import extract_audio_features_v1

SEED = 42
N_LAYERS = 1
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
BATCH_SIZE = 64
EPOCH_COUNT = 100


class CRNNClassifier:
    model: Model
    label_names = []
    monitor_metric = 'val_accuracy'

    seed: int = SEED
    n_layers: int = N_LAYERS
    filter_length: int = FILTER_LENGTH
    conv_filter_count: int = CONV_FILTER_COUNT
    epoch_count: int = EPOCH_COUNT

    def __init__(self, model: Model):
        self.model = model

    @classmethod
    def label_indexes_to_names(cls, label_indexes: Iterable[int]) -> List[str]:
        return list(map(lambda g: cls.label_names[g], label_indexes))

    def predict(self, audio_filepaths: List[str]) -> List[str]:

        if not audio_filepaths:
            return []

        first_audio_filepath = audio_filepaths[0]
        first_audio_features = extract_audio_features_v1(first_audio_filepath)

        features_shape = first_audio_features.shape
        extracted = [first_audio_features]
        for audio_filepath in audio_filepaths[1:]:
            extracted.append(extract_audio_features_v1(audio_filepath, features_shape))

        x = np.array(extracted)
        return self.label_indexes_to_names(np.argmax(self.model.predict(x), axis=-1))

    def predict_raw(self, X):
        return self.label_indexes_to_names(np.argmax(self.model.predict(X), axis=-1))

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
        return model

    @classmethod
    def train_model(cls,
                    model: Model,
                    X: np.array,
                    y: np.array,
                    output_path: str,
                    epochs: int = EPOCH_COUNT,
                    test_size=0.3
                    ):
        (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=test_size, random_state=SEED)

        monitor_metric = cls.monitor_metric
        print('Training...')
        history = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
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
    def train(cls,
              X: np.array,
              y: np.array,
              model_path: str,
              epochs: int = EPOCH_COUNT
              ):
        model = cls.build_model(X.shape[2])
        plot_model(model, show_shapes=True)
        model, history = cls.train_model(X, y, model, model_path, epochs)
        show_history(history, cls.__class__.__name__)
        return cls(model)

    @classmethod
    def load(cls, model_path: str):
        return cls(load_model(model_path))

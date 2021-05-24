import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.python.autograph import do_not_convert
from tensorflow.python.keras.utils.vis_utils import plot_model

from app.classifiers.abstract import AbstractClassifier

BATCH_SIZE = 128
EPOCHS = 400
SEED = 42


class LstmClassifier(AbstractClassifier):
    label_names = []
    monitor_metric = 'val_accuracy'
    epochs = EPOCHS

    @classmethod
    @do_not_convert
    def build_model(cls, n_features):
        n_labels = len(cls.label_names)

        input_shape = (None, n_features)
        model_input = Input(input_shape, name='input')
        layer = model_input
        layer = LSTM(
            units=128,
            dropout=0.05,
            recurrent_dropout=0.35,
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        )(layer)
        layer = LSTM(
            units=32,
            dropout=0.05,
            recurrent_dropout=0.35,
            return_sequences=False,
            name='lstm_2'
        )(layer)
        layer = Dense(
            units=n_labels,
            activation='softmax'
        )(layer)

        model_output = layer
        model = Model(model_input, model_output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy', TopKCategoricalAccuracy(3, name='top3-accuracy')]
        )
        return model

    @classmethod
    def train_model(cls,
                    model: Model,
                    X: np.array,
                    y: np.array,
                    output_path: str,
                    epochs: int = EPOCHS,
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
    def build_and_train(cls,
                        X: np.array,
                        y: np.array,
                        model_path: str,
                        epochs: int = EPOCHS
                        ):
        model = cls.build_model(X.shape[2])
        plot_model(model, show_shapes=True)
        # model, history = cls.train_model(model, X, y, model_path, epochs)
        # show_history(history, cls.__class__.__name__)
        return cls(model)

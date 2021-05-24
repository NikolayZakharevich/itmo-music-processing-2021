from os import PathLike
from typing import List, Iterable, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, GlobalAvgPool1D, Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.metrics import TopKCategoricalAccuracy
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

from app.classifiers.abstract import AbstractClassifier

N_LAYERS = 2
D_MODEL = 128
N_HEADS = 8
DFF = 256
MAXIMUM_POSITION_ENCODING = 2048

EPOCHS = 50


def top3_accuracy():
    return TopKCategoricalAccuracy(3, name='top3-accuracy')


class TransformerClassifier(AbstractClassifier):
    """
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, None, 128)]       0
    _________________________________________________________________
    encoder (Encoder)            (None, None, 128)         264960
    _________________________________________________________________
    dropout_5 (Dropout)          (None, None, 128)         0
    _________________________________________________________________
    global_average_pooling1d (Gl (None, 128)               0
    _________________________________________________________________
    dense_12 (Dense)             (None, 76)                9804
    _________________________________________________________________
    dense_13 (Dense)             (None, 19)                1463
    =================================================================
    Total params: 276,227
    Trainable params: 276,227
    Non-trainable params: 0
    """
    name: str = 'transformer_classifier'

    epochs: int = EPOCHS
    monitor_metric = 'val_accuracy'

    n_layers: int = N_LAYERS
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    dff: int = DFF
    maximum_position_encoding: int = MAXIMUM_POSITION_ENCODING

    @classmethod
    def label_indexes_to_names(cls, label_indexes: Iterable[int]) -> List[str]:
        return list(map(lambda g: cls.label_names[g], label_indexes))

    @classmethod
    def build_model(cls, n_features):
        n_classes = len(cls.label_names)

        inp = Input((None, cls.d_model))

        encoder = Encoder(
            n_layers=cls.n_layers,
            d_model=cls.d_model,
            n_heads=cls.n_heads,
            dff=cls.dff,
            maximum_position_encoding=cls.maximum_position_encoding,
            rate=0.3,
        )
        x = encoder(inp)
        x = Dropout(0.2)(x)
        x = GlobalAvgPool1D()(x)
        x = Dense(4 * n_classes, activation='selu')(x)
        out = Dense(n_classes, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out, name=cls.name)
        opt = Adam(0.00001)
        model.compile(
            optimizer=opt, loss=custom_binary_crossentropy, metrics=[custom_binary_accuracy, top3_accuracy()]
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
    ) -> tuple[Model, History]:
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
                    factor=0.9,
                    patience=10,
                    min_delta=0.01,
                    verbose=1
                )
            ]
        )
        return model, history

    @classmethod
    def load(cls, model_path: str):
        model = load_model(model_path, custom_objects={
            'custom_binary_crossentropy': custom_binary_crossentropy,
            'custom_binary_accuracy': custom_binary_accuracy,
            'Encoder': Encoder,
        })
        model.compile(loss=model.loss, optimizer=model.optimizer, metrics=[custom_binary_accuracy])
        return cls(model)


def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    y_true2 = math_ops.cast(y_true, y_pred.dtype)
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true2 = math_ops.cast(y_true2 > threshold, y_true2.dtype)

    return K.mean(math_ops.equal(y_true2, y_pred), axis=-1)


def custom_binary_crossentropy(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    epsilon_ = K._constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = clip_ops.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = 4 * y_true * math_ops.log(output + K.epsilon())
    bce += (1 - y_true) * math_ops.log(1 - output + K.epsilon())
    return K.sum(-bce, axis=-1)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    :param q:    query shape == (..., seq_len_q, depth)
    :param k:    key shape == (..., seq_len_k, depth)
    :param v:    value shape == (..., seq_len_v, depth_v)
    :param mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
    :return:      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):

    def __init__(
            self, n_layers, d_model, n_heads, dff, maximum_position_encoding, rate=0.1, **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, n_heads, dff, rate) for _ in range(n_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None, *args, **kwargs):
        seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.n_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate,
        })
        return config


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, name=None):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.name = name  # Modified from the source

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'name': self.name
        }

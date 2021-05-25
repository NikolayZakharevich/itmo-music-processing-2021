import os
from typing import Type, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from app.classifiers.abstract import AbstractClassifier
from app.common.columns import EMOTIONS, Column
from app.dataset.dumps import get_tracks_features_v1, get_single_track_features_path
from app.common.utils import cover_accuracy
from app.common.view import show_confusion_matrix
from app.dataset.dataset_tf import TracksGenerator
from app.emotion.classifiers import EmotionClassifierTransformer, EmotionClassifierCrnn, EmotionClassifierLstm
from classifiers.classifiers_torch import LstmClassifier, multiclass_train_lstm
from config import DIR_MODELS, FILE_TRACKS
from dataset.dataset_torch import multiclass_get_dataloaders_split
from features.features import N_MELS

MODEL_PATH_CRNN_V1 = DIR_MODELS + 'emotions_crnn_v1.h5'
MODEL_PATH_CRNN_V2 = DIR_MODELS + 'emotions_crnn_v2.h5'
MODEL_PATH_CRNN_V3 = DIR_MODELS + 'emotions_crnn_v3.h5'

MODEL_PATH_TRANSFORMER_V1 = DIR_MODELS + 'emotions_transformer_v1.h5'
MODEL_PATH_TRANSFORMER_V2 = DIR_MODELS + 'emotions_transformer_v2.h5'
MODEL_PATH_TRANSFORMER_V3 = DIR_MODELS + 'emotions_transformer_v3.h5'

MODEL_PATH_LSTM_V1 = DIR_MODELS + 'emotions_lstm_v1.h5'

TRACKS_TYPE_SINGLE_EMOTION_ONLY = 1
TRACKS_TYPE_MANY_EMOTIONS_ONLY = 2
TRACKS_TYPE_ALL = 3

N_FEATURES = 128


def get_track_ids_and_labels():
    filtered = []
    tracks = get_tracks(TRACKS_TYPE_SINGLE_EMOTION_ONLY)
    for _, track in tracks.iterrows():
        if os.path.isfile(get_single_track_features_path(track[Column.YM_TRACK_ID.value])):
            filtered.append(track)
    tracks = pd.DataFrame(filtered, columns=tracks.columns)

    tracks[Column.EMOTIONS.value] = tracks[Column.EMOTIONS.value].apply(lambda x: x.split('|'))

    track_ids = tracks[Column.YM_TRACK_ID.value].tolist()
    labels = tracks.set_index([Column.YM_TRACK_ID.value]).to_dict('dict')[Column.EMOTIONS.value]

    return track_ids, labels


def emotions_train_transformer_tf():
    emotions_train_tf(EmotionClassifierTransformer, MODEL_PATH_TRANSFORMER_V3)


def emotions_train_crnn_tf():
    emotions_train_tf(EmotionClassifierCrnn, MODEL_PATH_CRNN_V3)


def emotions_train_lstm_tf():
    emotions_train_tf(EmotionClassifierLstm, MODEL_PATH_LSTM_V1)


def emotions_train_lstm_torch():
    track_ids, labels = get_track_ids_and_labels()

    batch_size = 9
    dataloader_train, dataloader_val = multiclass_get_dataloaders_split(
        track_ids=track_ids,
        labels=labels,
        label_names=EMOTIONS,
        batch_size=9,
        test_split_size=0.1,
        test_split_seed=2021
    )

    input_dim = N_MELS
    hidden_dim = 32
    n_classes = len(EMOTIONS)
    n_layers = 2
    model = LstmClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        output_dim=n_classes,
        n_layers=n_layers
    )
    model.load_state_dict(torch.load('models/emotions-lstm_v2_top3-accuracy=0.4462.pt'))

    multiclass_train_lstm(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        filename_prefix='emotions-lstm'
    )


def emotions_train_tf(classifier: Type[AbstractClassifier], model_path: str, epochs: Optional[int] = None):
    filtered = []
    tracks = get_tracks(TRACKS_TYPE_SINGLE_EMOTION_ONLY)
    for _, track in tracks.iterrows():
        if os.path.isfile(get_single_track_features_path(track[Column.YM_TRACK_ID.value])):
            filtered.append(track)
    tracks = pd.DataFrame(filtered, columns=tracks.columns)

    tracks[Column.EMOTIONS.value] = tracks[Column.EMOTIONS.value].apply(lambda x: x.split('|'))

    track_ids = tracks[Column.YM_TRACK_ID.value].tolist()
    labels = tracks.set_index([Column.YM_TRACK_ID.value]).to_dict('dict')[Column.EMOTIONS.value]

    model = classifier.build_model(N_FEATURES)
    classifier.train_model(
        model=model,
        track_ids=track_ids,
        labels=labels,
        output_path=model_path,
        epochs=epochs,
        tracks_generator_class=TracksGenerator
    )


def emotions_test():
    clf = EmotionClassifierTransformer.load(MODEL_PATH_TRANSFORMER_V1)
    X_list, y_labels = load_data(get_tracks(TRACKS_TYPE_SINGLE_EMOTION_ONLY))
    X = np.array(X_list)
    y_pred_labels = clf.predict_top_1(X)
    show_confusion_matrix(y_labels, y_pred_labels, EMOTIONS, figsize=(15, 15))

    X_list, y_labels = load_data(get_tracks(TRACKS_TYPE_SINGLE_EMOTION_ONLY))
    X = np.array(X_list)
    y_pred_labels_top3 = clf.predict_top_k(X)
    print('Top3 accuracy: ', cover_accuracy(y_labels, y_pred_labels_top3))


def export_to_toloka_check():
    tracks = get_tracks()
    audio_paths = tracks.to_dict('dict')[Column.AUDIO_PATH.value]
    track_ids, X_list, y_labels = load_data_with_ids(tracks)
    X = np.array(X_list)

    model = EmotionClassifierCrnn.load(MODEL_PATH_CRNN_V1)

    y_pred = model.predict_top_1(X)
    y_pred_labels = multihot_inverse_transform(y_pred)

    data = []
    for i in range(len(y_labels)):
        if y_labels[i] != y_pred[i]:
            track_id = track_ids[i]
            audio_path = audio_paths[track_id]
            print({
                'true label': y_labels[i],
                'pred label:': y_pred_labels[i],
                'audio url': 'https://sky4uk.xyz/static/' + audio_path
            })
            data.append([y_pred_labels[i], audio_path])
    columns = ['emotion', 'audio_path']
    df = pd.DataFrame(data=data, columns=columns)
    df.drop_duplicates(subset=columns, inplace=True)
    df.to_csv('test.csv', index=False)


def load_data_with_ids(tracks: pd.DataFrame) -> tuple[list[int], list[np.ndarray], list[list[str]]]:
    tracks_features = get_tracks_features_v1(set(tracks[Column.YM_TRACK_ID.value]))

    track_ids, X_list, y_labels_all = [], [], []
    for _, track in tracks.iterrows():
        track_id = track[Column.YM_TRACK_ID.value]
        emotions = str(track[Column.EMOTIONS.value]).split('|')

        features_chunks = tracks_features.get(track_id, None)
        if not features_chunks:
            print(f'Missing audio features for track with id={track_id}')
            continue

        for features_chunk in features_chunks:
            track_ids.append(track_id)
            X_list.append(features_chunk)
            y_labels_all.append(emotions)

    return track_ids, X_list, y_labels_all


def load_data(tracks: pd.DataFrame) -> tuple[list[np.ndarray], list[list[str]]]:
    tracks_features = get_tracks_features_v1(set(tracks[Column.YM_TRACK_ID.value]))

    X_list, y_labels_all = [], []
    for _, track in tracks.iterrows():
        track_id = track[Column.YM_TRACK_ID.value]
        emotions = str(track[Column.EMOTIONS.value]).split('|')

        features_chunks = tracks_features.get(track_id, None)
        if not features_chunks:
            print(f'Missing audio features for track with id={track_id}')
            continue

        for features_chunk in features_chunks:
            X_list.append(features_chunk)
            y_labels_all.append(emotions)

    return X_list, y_labels_all


def get_tracks(tracks_type: int = TRACKS_TYPE_MANY_EMOTIONS_ONLY) -> pd.DataFrame:
    tracks = pd.read_csv(FILE_TRACKS)
    tracks.dropna(subset=[Column.EMOTIONS.value], inplace=True)

    def predicate(t) -> bool:
        emotions = t[Column.EMOTIONS.value]
        if tracks_type == TRACKS_TYPE_MANY_EMOTIONS_ONLY:
            return '|' in emotions
        if tracks_type == TRACKS_TYPE_SINGLE_EMOTION_ONLY:
            return '|' not in emotions
        if tracks_type == TRACKS_TYPE_ALL:
            return True
        return True

    tracks = tracks[tracks.apply(predicate, axis=1)]
    # show_emotions_frequencies(tracks)
    return tracks


def multihot_transform(y_labels: list[list[str]]) -> np.ndarray:
    mlb = MultiLabelBinarizer()
    mlb.fit([EMOTIONS])
    return mlb.transform(y_labels)


def multihot_inverse_transform(y: np.ndarray) -> list[list[str]]:
    mlb = MultiLabelBinarizer()
    mlb.fit([EMOTIONS])
    return mlb.inverse_transform(y)

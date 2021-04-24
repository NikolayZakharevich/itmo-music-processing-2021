import os
from typing import Union

import pandas as pd

from app.common.columns import EMOTIONS, Column
from app.common.view import display_confusion_matrix
from app.emotion.crnn import EmotionClassifier
from app.features.features import load_data
from app.genre.crnn import GenreClassifier, GENRES

DIR_DATA = 'data/'
DIR_AUDIOS = DIR_DATA + 'audios/'

FILE_TRACKS = DIR_DATA + 'tracks.csv'

DIR_GTZAN = DIR_DATA + 'gtzan/'
DIR_GTZAN_GENRES_ORIGINAL = DIR_GTZAN + 'genres_original/'

DIR_DUMPED = DIR_DATA + 'dumped/'
DIR_DUMPED_EMOTIONS = DIR_DUMPED + 'emotions-v3/'
DATA_GENRE_GTZAN_V1 = DIR_DUMPED + 'data-genre-gtzan-v1.pkl'
DATA_EMOTIONS_ALL = DIR_DUMPED + 'data-emotions-all.pkl'
DATA_EMOTIONS_V1 = DIR_DUMPED + 'data-emotions-v1.pkl'
DATA_EMOTIONS_V2 = DIR_DUMPED + 'data-emotions-v2.pkl'
DATA_EMOTIONS_V3 = DIR_DUMPED + 'data-emotions-v3.pkl'
DATA_EMOTIONS_TEST_V1 = DIR_DUMPED + 'data-emotions-test-v1.pkl'

DIR_MODELS = 'models/'
MODEL_PATH_CRNN_EMOTION_CLASSIFIER_V1 = DIR_MODELS + 'crnn_emotion_classifier_v1.h5'
MODEL_PATH_CRNN_EMOTION_CLASSIFIER_V2 = DIR_MODELS + 'crnn_emotion_classifier_v2.h5'
MODEL_PATH_CRNN_EMOTION_CLASSIFIER_V3 = DIR_MODELS + 'crnn_emotion_classifier_v3.h5'
MODEL_PATH_CRNN_EMOTION_CLASSIFIER_MINI = DIR_MODELS + 'crnn_emotion_classifier_mini.h5'

MODEL_PATH_CRNN_GENRE_CLASSIFIER_V1 = DIR_MODELS + 'crnn_genre_classifier_v1.h5'

MODEL_PATH_LSTM_EMOTION_CLASSIFIER_OLD = DIR_MODELS + 'lstm_emotion_classifier_old.h5'


def test_genre_classifier_my():
    tracks = pd.read_csv(FILE_TRACKS)
    tracks = tracks[tracks[Column.YM_GENRE.value].isin(GENRES)]
    batch = tracks.groupby('genre').head(1)

    clf = GenreClassifier.load(MODEL_PATH_CRNN_GENRE_CLASSIFIER_V1)
    labels = batch.genre.tolist()
    predictions = clf.predict(batch.audio_path.tolist())
    display_confusion_matrix(labels, predictions, GENRES)


def test_genre_classifier_gtzan():
    model_path = MODEL_PATH_CRNN_GENRE_CLASSIFIER_V1

    audio_paths = []
    labels = []
    for dir_genre in os.listdir(DIR_GTZAN_GENRES_ORIGINAL):
        if dir_genre not in GENRES:
            continue

        for audio_path in os.listdir(DIR_GTZAN_GENRES_ORIGINAL + dir_genre):
            audio_paths.append(DIR_GTZAN_GENRES_ORIGINAL + dir_genre + "/" + audio_path)
            labels.append(dir_genre)

    clf = GenreClassifier.load(model_path)
    predictions = clf.predict(audio_paths)
    display_confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        labels=GENRES,
        title='GTZAN'
    )


def train_genre_classifier_crnn_gtzan():
    input_data_path = DATA_GENRE_GTZAN_V1
    output_model_path = MODEL_PATH_CRNN_GENRE_CLASSIFIER_V1

    data = load_data(input_data_path)
    return GenreClassifier.train(data['x'], data['y'], output_model_path, epochs=100)


def train_emotion_classifier(
        output_path: str,
        dumped_data_path: Union[str, None] = None
) -> EmotionClassifier:
    data = load_data(dumped_data_path)
    x, y = data['x'], data['y']
    return EmotionClassifier.train(x, y, output_path, epochs=4)


def test_emotion_classifier(model_path: str):
    tracks = pd.read_csv(FILE_TRACKS).head(10)

    clf = EmotionClassifier.load(model_path)
    labels = tracks.emotion.tolist()
    predictions = clf.predict(tracks.audio_path.tolist())
    display_confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        labels=EMOTIONS,
        title='Мой датасет',
        figsize=(20, 20)
    )


def train_and_test_emotion_classifier():
    model_path = MODEL_PATH_CRNN_EMOTION_CLASSIFIER_V3
    dump_path = DATA_EMOTIONS_V2

    train_emotion_classifier(
        output_path=model_path,
        dumped_data_path=dump_path
    )
    # test_emotion_classifier(model_path)


def train_emotion_classifier_2(
        output_path: str,
        dumped_data_path: Union[str, None] = None
) -> EmotionClassifier:
    from app.common.crnn_v2 import EmotionClassifier2

    data = load_data(dumped_data_path)
    x = data['x']
    y = data['y']
    return EmotionClassifier2.train(x, y, output_path, epochs=4)


if __name__ == '__main__':
    pass

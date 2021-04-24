from app.common.crnn import CRNNClassifier

GENRES = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]


class GenreClassifier(CRNNClassifier):
    label_names = GENRES

from app.classifiers.crnn import CrnnClassifier

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


class GenreClassifier(CrnnClassifier):
    label_names = GENRES

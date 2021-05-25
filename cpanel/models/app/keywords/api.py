import collections

import pandas as pd
import torch

from classifiers.classifiers_torch import MultilabelLstmClassifier, multilabel_train_lstm
from config import FILE_KEYWORDS
from dataset.dataset_torch import multilabel_get_dataloaders_split
from dataset.dumps import dump_file_exists
from features.features import N_MELS


def save_classes(classes: list[str]):
    with open('keyword-classes.txt', 'w') as f:
        f.write(','.join(classes))


def keywords_train_lstm_torch():
    track_ids, labels, label_names = get_training_data()

    batch_size = 9
    dataloader_train, dataloader_val = multilabel_get_dataloaders_split(
        track_ids=track_ids,
        labels=labels,
        label_names=label_names,
        batch_size=9,
        test_split_size=0.1,
        test_split_seed=2021
    )

    input_dim = N_MELS
    hidden_dim = 32
    n_classes = len(label_names)
    n_layers = 3

    model = MultilabelLstmClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=n_classes,
        batch_size=batch_size,
        n_layers=n_layers
    )
    model.load_state_dict(torch.load('models/keywords_v2_loss=-0.1986.pt'))

    multilabel_train_lstm(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        filename_prefix='keywords'
    )


def get_training_data() -> tuple[list[int], dict[int, list[str]], list[str]]:
    """
    >>> track_ids, labels, label_names = get_training_data()
    :return:
    """
    keywords_df = pd.read_csv(FILE_KEYWORDS)

    c = collections.Counter()

    tracks_keywords = {}
    for _, row in keywords_df.iterrows():
        track_id = row.track_id
        if not dump_file_exists(track_id):
            continue

        keywords = row.keywords.split(';')
        weights = list(map(float, row.weights.split(';')))

        tracks_keywords[row.track_id] = []
        for i in range(len(keywords)):
            tracks_keywords[row.track_id].append(keywords[i])
            c[keywords[i]] += 1

    classes: list[str] = list(map(lambda x: x[0], c.most_common(50)))
    save_classes(classes)

    filtered: dict[int, list[str]] = {}
    for track_id, keywords in tracks_keywords.items():
        filtered_keywords: list[str] = list(filter(lambda k: k in classes, keywords))
        if len(filtered_keywords) > 0:
            filtered[track_id] = filtered_keywords

    return list(filtered.keys()), filtered, classes

import torch
from ignite.utils import manual_seed

from app.classifiers.classifiers_torch import train_lstm, LstmClassifier
from app.common.columns import EMOTIONS
from app.dataset.dataset_torch import get_dataloaders_split
from app.emotion.api import get_track_ids_and_labels
from app.features.features import N_MELS

EPS = 1e-7
SEED = 1234
manual_seed(SEED)

if __name__ == '__main__':
    track_ids, labels = get_track_ids_and_labels()

    batch_size = 9
    dataloader_train, dataloader_val = get_dataloaders_split(
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
    model.load_state_dict(torch.load('models/pytorch-emotions_pytorch-emotions_8080.pt'))

    train_lstm(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        filename_prefix='emotions-lstm'
    )

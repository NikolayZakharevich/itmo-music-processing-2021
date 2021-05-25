from collections import Counter
from os import PathLike
from typing import Union, List

import numpy as np
import torch
from torch import nn

from app.features import N_MELS

MODEL_PATH = 'models/keywords.pt'

keywords = [
    'love', 'heart', 'way', 'time', 'eyes', 'life', 'night', 'baby', 'everything', 'mind', 'world', 'nothing',
    'something', 'day', 'things', 'soul', 'sun', 'one', 'home', 'pain', 'hands', 'dreams', 'end', 'god', 'dream',
    'place', 'head', 'words', 'yeah', 'someone', 'sky', 'cause', 'girl', 'moment', 'hand', 'truth', 'man', 'fire',
    'face', 'light', 'tears', 'ooh', 'name', 'song', 'somebody', 'side', 'morning', 'smile', 'nobody', 'tonight'
]


class KeywordsSuggester(nn.Module):
    """
    KeywordsSuggester
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            batch_size: int = 9,
            n_layers: int = 3
    ):
        """
        :param input_dim: The number of expected features in the input `x`
        :param hidden_dim: The number of features in the hidden state `h`
        :param batch_size:
        :param output_dim:
        :param n_layers:
        """
        super(KeywordsSuggester, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.output = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        logits = self.output(lstm_out[:, -1])
        return logits


def load_model(model_path: Union[str, bytes, PathLike]) -> nn.Module:
    """
    Loads model from state dict
    :param model_path:
    :return:
    """
    input_dim = N_MELS
    hidden_dim = 32
    n_classes = len(keywords)

    model = KeywordsSuggester(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=n_classes
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_keywords(features: np.ndarray, k=10) -> List[str]:
    model = load_model(MODEL_PATH)
    output = model(torch.tensor(features))
    indices = torch.flatten(torch.topk(output, k, dim=1)[1])
    indices = list(map(lambda x: x[0], Counter(indices).most_common(k)))
    return [keywords[i] for i in indices]

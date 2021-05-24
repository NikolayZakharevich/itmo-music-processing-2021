import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import CrossEntropyLoss


#
# Data formatters
#

def to_prediction_top1(output: Tensor) -> Tensor:
    """
    :param output:
    :return: one-hot representation of output
    """
    num_classes = output.size()[1]
    return F.one_hot(output.argmax(dim=1), num_classes=num_classes)


def to_prediction_top1_le(output: Tensor) -> Tensor:
    """
    :param output:
    :return: one-hot representation of output
    """
    return output.argmax(dim=1)


def to_prediction_topk(output: Tensor, k: int = 3) -> Tensor:
    """
    :param k:
    :param output:
    :return: multi-hot representation of output
    """
    result = torch.zeros(output.size(), dtype=torch.int64)
    _, indices = torch.topk(output, k, dim=1)
    for i in range(output.size()[0]):
        result[i] = result[i].index_fill(0, indices[i], 1)
    return result


def to_prediction_topk_le(output: Tensor, k: int = 3) -> Tensor:
    """
    le — label encoded
    :param k:
    :param output:
    :return: multi-hot representation of output

    >>> to_prediction_topk_le(Tensor([[0.1, 0.2, 0.7, 0.5], [0.5, 0.5, 0.0, 0.0]]), k=2)
    torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
    """
    return torch.topk(output, k, dim=1)[1]


#
# Loss functions
#

class CrossEntropyLossOneHot(CrossEntropyLoss):
    EPS: int = 1e-7

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        input = torch.clip(input, self.EPS, 1 - self.EPS)
        crossentropy = target * torch.log(input)
        crossentropy = -torch.sum(crossentropy, -1)
        return torch.sum(crossentropy)


class CrossEntropyLossLe(CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


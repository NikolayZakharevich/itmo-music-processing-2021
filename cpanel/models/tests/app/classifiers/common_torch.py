import unittest

import torch
from torch import Tensor

from app.classifiers.common_torch import to_prediction_top1, to_prediction_topk, to_prediction_top1_le, \
    to_prediction_topk_le
from tests.common import data_provider

import ignite.engine  # ignite import bug workaround
from ignite.metrics import TopKCategoricalAccuracy


class TestOneHotTransformers(unittest.TestCase):
    outputs_top1 = lambda: (
        (
            Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]]),
            torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.long)
        ),
        (
            Tensor([[0.3, 0.5, 0.2]]),
            torch.tensor([[0, 1, 0]], dtype=torch.long)
        )
    )

    outputs_top1_le = lambda: (
        (
            Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]]),
            torch.tensor([2, 0], dtype=torch.long)
        ),
        (
            Tensor([[0.3, 0.5, 0.2]]),
            torch.tensor([1], dtype=torch.long)
        )
    )

    outputs_topk = lambda: (
        (
            2,
            Tensor([[0.1, 0.2, 0.7, 0.5], [0.5, 0.5, 0.0, 0.0]]),
            torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.long)
        ),
        (
            3,
            Tensor([[0.3, 0.5, 0.2]]),
            torch.tensor([[1, 1, 1]], dtype=torch.long)
        )
    )

    outputs_topk_le = lambda: (
        (
            2,
            Tensor([[0.1, 0.2, 0.7, 0.5], [0.5, 0.5, 0.0, 0.0]]),
            torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
        ),
        (
            3,
            Tensor([[0.3, 0.5, 0.2]]),
            torch.tensor([[0, 1, 2]], dtype=torch.long)
        )
    )

    @data_provider(outputs_top1)
    def test_to_prediction_top1(self, softmax_output: Tensor, expected_prediction: Tensor):
        lhs = to_prediction_top1(softmax_output)
        self.assertTrue(torch.equal(lhs, expected_prediction))

    @data_provider(outputs_top1_le)
    def test_to_prediction_top1_le(self, softmax_output: Tensor, expected_prediction: Tensor):
        lhs = to_prediction_top1_le(softmax_output)
        self.assertTrue(torch.equal(lhs, expected_prediction))

    @data_provider(outputs_topk)
    def test_to_prediction_topk(self, k: int, softmax_output: Tensor, expected_prediction: Tensor):
        lhs = to_prediction_topk(softmax_output, k=k)
        self.assertTrue(torch.equal(lhs, expected_prediction))

    @data_provider(outputs_topk_le)
    def test_to_prediction_topk_le(self, k: int, softmax_output: Tensor, expected_prediction: Tensor):
        lhs = set(torch.reshape(to_prediction_topk_le(softmax_output, k=k), (-1,)))
        rhs = set(torch.reshape(expected_prediction, (-1,)))
        self.assertTrue(lhs, rhs)


class TestCustomMetrics(unittest.TestCase):
    outputs = lambda: (
        (
            2,
            torch.tensor([[0.1, 0.2, 0.7, 0.5], [0.5, 0.5, 0.0, 0.0]]),
            torch.tensor([[2], [0]], dtype=torch.long),
            1.0
        ),
        (
            1,
            torch.tensor([[0.1, 0.2, 0.7, 0.5], [0.5, 0.5, 0.0, 0.0]]),
            torch.tensor([[2], [2]], dtype=torch.long),
            0.5
        )
    )

    @data_provider(outputs)
    def test_topk_accuracy(self, k: int, y_pred: Tensor, y_true: Tensor, score: float):
        accuracy = TopKCategoricalAccuracy(k=k)
        accuracy.update((y_pred, y_true))
        self.assertEqual(score, accuracy.compute())


if __name__ == '__main__':
    unittest.main()

import unittest

from app.common.columns import Emotion, EMOTIONS
from dataset.dataset_torch import get_dataloaders_split
from tests.common import data_provider


class TestDataset(unittest.TestCase):
    dataset_track_ids = lambda: (
        (
            [79909359], {79909359: Emotion.HAPPY.value}
        ),
        (
            [79909359, 79909360, 79909362, 79909363, 79909364],
            {79909359: Emotion.HAPPY.value,
             79909360: Emotion.FUNNY.value,
             79909362: Emotion.SERIOUS.value,
             79909363: Emotion.HAPPY.value,
             79909364: Emotion.SERIOUS.value}
        )
    )

    @data_provider(dataset_track_ids)
    def test_get_dataset(self, track_ids: list[int], labels: dict[int, str]):
        d1, d2 = get_dataloaders_split(
            track_ids=track_ids,
            labels=labels,
            label_names=EMOTIONS
        )
        self.assertIsNotNone(d1)
        self.assertIsNotNone(d2)


if __name__ == '__main__':
    unittest.main()

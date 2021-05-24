import os
import unittest

import librosa
import torch
import torchaudio
from torch import Tensor
from torchaudio.transforms import MelSpectrogram

from app.common.tracks import get_audio_path_default
from app.features.features import WINDOW_SIZE, WINDOW_STRIDE, N_MELS, MEL_KWARGS, SAMPLE_RATE
from tests.common import data_provider


class TestFeatureExtraction(unittest.TestCase):
    dataset_track_ids = lambda: (
        (
            torch.tensor([[79909359]], dtype=torch.long),
        ),
        (
            torch.tensor([[79909359, 79909360, 79909363, 79909364]], dtype=torch.long),
        )
    )

    @data_provider(dataset_track_ids)
    def test_melspectrogram(self, track_ids: Tensor):
        def file_exists(track_id: int) -> bool:
            return os.path.isfile(get_audio_path_default(track_id))

        def get_melspectrogram_torchaudio(track_id: int) -> Tensor:
            path = get_audio_path_default(track_id)
            effects = [
                ['remix', '2'],
                ['rate', str(SAMPLE_RATE)],
            ]
            waveform, _ = torchaudio.sox_effects.apply_effects_file(path, effects)
            return transform(waveform)[0]

        def get_melspectrogram_librosa(track_id: int) -> Tensor:
            new_input, sample_rate = librosa.load(get_audio_path_default(track_id))
            return torch.tensor(librosa.feature.melspectrogram(new_input, **MEL_KWARGS))

        transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WINDOW_SIZE,
            hop_length=WINDOW_STRIDE,
            n_mels=N_MELS
        )

        for batch in track_ids:
            for track_id in batch:
                if not file_exists(track_id):
                    continue
                melspectrogram_torchaudio = get_melspectrogram_torchaudio(track_id)
                melspectrogram_librosa = get_melspectrogram_librosa(track_id)
                self.assertEqual(melspectrogram_torchaudio.size(), melspectrogram_librosa.size())


if __name__ == '__main__':
    unittest.main()

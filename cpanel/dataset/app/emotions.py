import numpy as np

from app.columns import Emotion


def get_emotion_embedding(emotion: Emotion) -> np.ndarray:
    return np.zeros(2)


def get_k_closest_emotions(emotion: Emotion, k: int = 3) -> list[Emotion]:
    return []

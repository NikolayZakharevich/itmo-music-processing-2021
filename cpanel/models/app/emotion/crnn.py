from app.common.columns import EMOTIONS
from app.common.crnn import CRNNClassifier


class EmotionClassifier(CRNNClassifier):
    label_names = EMOTIONS

    monitor_metric = 'val_top3-accuracy'

from app.common.columns import EMOTIONS
from app.classifiers.crnn import CrnnClassifier
from app.classifiers.lstm import LstmClassifier
from app.classifiers.transformer_tf import TransformerClassifier


class EmotionClassifierCrnn(CrnnClassifier):
    """
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           [(None, None, 128)]       0
    _________________________________________________________________
    convolution_1 (Conv1D)       (None, None, 256)         164096
    _________________________________________________________________
    batch_normalization (BatchNo (None, None, 256)         1024
    _________________________________________________________________
    activation (Activation)      (None, None, 256)         0
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, None, 256)         0
    _________________________________________________________________
    dropout (Dropout)            (None, None, 256)         0
    _________________________________________________________________
    time_distributed (TimeDistri (None, None, 19)          4883
    _________________________________________________________________
    output_merged (Lambda)       (None, 19)                0
    _________________________________________________________________
    output_realtime (Activation) (None, 19)                0
    =================================================================
    Total params: 170,003
    Trainable params: 169,491
    Non-trainable params: 512
    _________________________________________________________________
    """
    label_names = EMOTIONS
    monitor_metric = 'val_top3-accuracy'


class EmotionClassifierLstm(LstmClassifier):
    label_names = EMOTIONS
    monitor_metric = 'val_top3-accuracy'


class EmotionClassifierTransformer(TransformerClassifier):
    label_names = EMOTIONS
    monitor_metric = 'val_loss'

import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, TopKCategoricalAccuracy, RunningAverage, Loss
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

import torch

from app.classifiers.common_torch import CrossEntropyLossLe, CrossEntropyLossOneHot
from config import DIR_MODELS


class LstmClassifier(nn.Module):
    """
    LSTM Classifier
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            batch_size: int = 9,
            n_layers: int = 2
    ):
        """
        :param input_dim: The number of expected features in the input `x`
        :param hidden_dim: The number of features in the hidden state `h`
        :param batch_size:
        :param output_dim:
        :param n_layers:
        """
        super(LstmClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.output = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        logits = self.output(lstm_out[:, -1])
        return F.softmax(logits, dim=1)


def multiclass_train_lstm(
        model: LstmClassifier,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        filename_prefix: str,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = CrossEntropyLossOneHot()

    def process_function(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return y_pred, y, loss.item(),

    def eval_function(_engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y = y.to(device)
            x = x.to(device)
            y_pred = model(x)
            return y_pred, y

    def score_function(engine):
        return engine.state.metrics['top3-accuracy']

    model.to(device)

    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)

    accuracy_top1 = Accuracy(output_transform=lambda x: (x[0], x[1]), device=device, is_multilabel=True)
    accuracy_top3 = TopKCategoricalAccuracy(output_transform=lambda x: (x[0], x[1]), k=3, device=device)

    RunningAverage(accuracy_top1).attach(trainer, 'accuracy')
    RunningAverage(accuracy_top3).attach(trainer, 'top3-accuracy')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss')

    accuracy_top1.attach(train_evaluator, 'accuracy')
    accuracy_top3.attach(train_evaluator, 'top3-accuracy')
    Loss(criterion).attach(train_evaluator, 'loss')

    accuracy_top1.attach(validation_evaluator, 'accuracy')
    accuracy_top3.attach(validation_evaluator, 'top3-accuracy')
    Loss(criterion).attach(validation_evaluator, 'loss')

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(
        engine=trainer,
        metric_names='all'
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(dataloader_train)
        message = f'Training results - Epoch: {engine.state.epoch}.'
        for metric_name, score in train_evaluator.state.metrics.items():
            message += f' {metric_name}: {score:.2f}.'
        pbar.log_message(message)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(dataloader_val)
        message = f'Validation results - Epoch: {engine.state.epoch}.'
        for metric_name, score in train_evaluator.state.metrics.items():
            message += f' {metric_name}: {score:.2f}.'
        pbar.log_message(message)
        pbar.n = pbar.last_print_n = 0

    validation_evaluator.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    )

    checkpointer = ModelCheckpoint(
        dirname=DIR_MODELS,
        filename_prefix=filename_prefix,
        score_function=score_function,
        score_name='top3-accuracy',
        n_saved=2,
        create_dir=True,
        save_as_state_dict=True,
        require_empty=False
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'v2': model})

    trainer.run(dataloader_train, max_epochs=20)


class MultilabelLstmClassifier(nn.Module):
    """
    LSTM Classifier
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            batch_size: int = 9,
            n_layers: int = 2
    ):
        """
        :param input_dim: The number of expected features in the input `x`
        :param hidden_dim: The number of features in the hidden state `h`
        :param batch_size:
        :param output_dim:
        :param n_layers:
        """
        super(MultilabelLstmClassifier, self).__init__()
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


def multilabel_train_lstm(
        model: MultilabelLstmClassifier,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        filename_prefix: str,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = BCEWithLogitsLoss()

    def process_function(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
        return y_pred, y, loss.item(),

    def eval_function(_engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y = y.to(device)
            x = x.to(device)
            y_pred = model(x)
            return y_pred, y.float()

    def score_function(engine):
        return -engine.state.metrics['loss']

    model.to(device)

    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)

    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss')
    Loss(criterion).attach(train_evaluator, 'loss')
    Loss(criterion).attach(validation_evaluator, 'loss')

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(engine=trainer, metric_names='all')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(dataloader_train)
        message = f'Training results - Epoch: {engine.state.epoch}.'
        for metric_name, score in train_evaluator.state.metrics.items():
            message += f' {metric_name}: {score:.2f}.'
        pbar.log_message(message)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(dataloader_val)
        message = f'Validation results - Epoch: {engine.state.epoch}.'
        for metric_name, score in train_evaluator.state.metrics.items():
            message += f' {metric_name}: {score:.2f}.'
        pbar.log_message(message)
        pbar.n = pbar.last_print_n = 0

    validation_evaluator.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    )

    checkpointer = ModelCheckpoint(
        dirname=DIR_MODELS,
        filename_prefix=filename_prefix,
        score_function=score_function,
        score_name='loss',
        n_saved=2,
        create_dir=True,
        save_as_state_dict=True,
        require_empty=False
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'v2': model})

    trainer.run(dataloader_train, max_epochs=20)

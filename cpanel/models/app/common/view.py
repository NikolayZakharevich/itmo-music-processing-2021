import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score

from app.common.columns import Column


def show_confusion_matrix(
        y_true,
        y_pred,
        labels,
        title='',
        figsize=(10, 10)
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    title = title + '\naсcuracy: ' + str(accuracy_score(y_true, y_pred))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    cm_display.plot(ax=ax)
    plt.show()


def show_emotions_frequencies(tracks: pd.DataFrame):
    show_frequencies(
        tracks[Column.EMOTIONS.value].apply(lambda s: str(s).split('|')).explode(),
        'Частота эмоций'
    )


def show_frequencies(items: pd.Series, title: str = 'Частота'):
    label_freq = items.value_counts().sort_values(ascending=False)

    title = f'{title} (сумма по всем — {sum(label_freq)})'

    style.use('fivethirtyeight')
    plt.figure(figsize=(12, 10))
    sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index)
    plt.title(title, fontsize=14)
    plt.xlabel('')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def show_history(history, title):
    best_top3_accuracy = max(history.history['val_top3-accuracy'])

    style.use('bmh')

    plt.title(f'{title}: model accuracy\nBest top3-accuracy: {best_top3_accuracy}')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['top3-accuracy'])
    plt.plot(history.history['val_top3-accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(
        ['top-1 accuracy [train]', 'top-1 accuracy [validate]', 'top-3 accuracy [train]', 'top-3 accuracy [validate]'],
        loc='upper left')
    plt.show()

    # summarize history for loss
    plt.title(f'{title}: model loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'loss validation'], loc='lower left')
    plt.show()


def learning_curves(history):
    """Plot the learning curves of loss and macro f1 score
    for the training and validation datasets.

    Args:
        history: history callback of fitting a tensorflow keras model
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']

    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), macro_f1, label='Training Macro F1-score')
    plt.plot(range(1, epochs + 1), val_macro_f1, label='Validation Macro F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('Macro F1-score')
    plt.title('Training and Validation Macro F1-score')
    plt.xlabel('epoch')

    plt.show()

    return loss, val_loss, macro_f1, val_macro_f1

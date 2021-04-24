import pandas as pd
import matplotlib.pyplot as plt


def show_count(df: pd.DataFrame, title: str):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    title = f'{title}\nКоличество треков: {len(df)}'

    df.emotion.value_counts().plot.barh(
        ax=axes[0],
        figsize=(10, 10),
        title='Распределение по эмоциям',
    )
    df.genre.value_counts().plot.barh(
        ax=axes[1],
        figsize=(10, 10),
        title='Распределение по жанрам',
    )

    fig.suptitle(title)
    plt.show()

import os
from typing import Callable


def add_trailing_path_separator(path: str) -> str:
    return os.path.join(path, '')


def split_equal_chunks(l: list, chunk_size: int):
    """
    Ignores tail after last chunk
    >>> split_equal_chunks(list(range(10)), 3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> split_equal_chunks(list(range(10)), 2)
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    :param l:
    :param chunk_size:
    :return:
    """
    return [l[i - chunk_size: i] for i in range(chunk_size, len(l) + 1, chunk_size)]


def compose2(f: Callable, g: Callable) -> Callable:
    return lambda x: g(f(x))


def compose3(f: Callable, g: Callable, h: Callable) -> Callable:
    return lambda x: h(g(f(x)))


def cover_accuracy(y_true, y_pred):
    ok = 0
    for i in range(len(y_true)):
        ok_found = False
        for label_true in y_true[i]:
            if ok_found:
                break
            for label_pred in y_pred[i]:
                if label_true == label_pred:
                    ok_found = True
                    ok += 1
                    break
    return ok / len(y_true)

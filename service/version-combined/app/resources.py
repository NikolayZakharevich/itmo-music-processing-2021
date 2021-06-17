import json
import os

DIR_RESOURCES = 'resources'

PATH_EMOTION_FONTS = os.path.join(DIR_RESOURCES, 'emotion_fonts.json')
PATH_KEYWORDS = os.path.join(DIR_RESOURCES, 'keywords.txt')


def load_keywords() -> list[str]:
    with open(PATH_KEYWORDS) as file_keywords:
        return file_keywords.read().split(';')


def load_emotion_fonts() -> dict[str, list[str]]:
    with open(PATH_EMOTION_FONTS) as file_emotion_fonts:
        return json.load(file_emotion_fonts)

import json
import os
from typing import List, Dict

DIR_RESOURCES = 'resources'

PATH_EMOTION_FONTS = os.path.join(DIR_RESOURCES, 'emotion_fonts.json')
PATH_KEYWORDS = os.path.join(DIR_RESOURCES, 'keywords.txt')


def load_keywords() -> List[str]:
    with open(PATH_KEYWORDS) as file_keywords:
        return file_keywords.read().split(';')


def load_emotion_fonts() -> Dict[str, List[str]]:
    with open(PATH_EMOTION_FONTS) as file_emotion_fonts:
        return json.load(file_emotion_fonts)

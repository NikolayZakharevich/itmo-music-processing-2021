import json
import os
from typing import Dict, List

DIR_RESOURCES = 'resources'

PATH_EMOTION_FONTS = os.path.join(DIR_RESOURCES, 'emotion_fonts.json')


def load_emotion_fonts() -> Dict[str, List[str]]:
    with open(PATH_EMOTION_FONTS) as file_emotion_fonts:
        return json.load(file_emotion_fonts)

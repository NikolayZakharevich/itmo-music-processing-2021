from dotenv import load_dotenv
import os

from app.common.utils import add_trailing_path_separator

load_dotenv()


def to_absolute_path(relative_path: str) -> str:
    return DIR_BASE + relative_path


DIR_BASE = add_trailing_path_separator(os.getenv('DIR_BASE', os.getcwd()))

DIR_DATA = add_trailing_path_separator(os.getenv('DIR_DATA', '../dataset/data'))
DIR_MODELS = 'models/'
DIR_AUDIOS = DIR_DATA + 'audios/'
DIR_FEATURES_V1 = DIR_DATA + 'features_v1/'
DIR_FEATURES_V1_SINGLE = DIR_DATA + 'features_v1_single/'

FILE_TRACKS = DIR_DATA + 'tracks.csv'

DUMP_FILE_EXTENSION_BATCH = 'pkl'
DUMP_FILE_EXTENSION_SINGLE = 'npy'

FILE_KEYWORDS = DIR_DATA + 'keywords.csv'

STORAGE_URL_STATIC = os.getenv('STORAGE_URL_STATIC')

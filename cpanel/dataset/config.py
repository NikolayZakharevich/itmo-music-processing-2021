from dotenv import load_dotenv
import os

load_dotenv()

TOKEN_YANDEX_MUSIC = os.getenv('TOKEN_YANDEX_MUSIC')
TOKEN_YANDEX_TOLOKA = os.getenv('TOKEN_YANDEX_TOLOKA')

YANDEX_TOLOKA_PROJECT_ID = os.getenv('YANDEX_TOLOKA_PROJECT_ID')
YANDEX_TOLOKA_TRAINING_POOL_ID = os.getenv('YANDEX_TOLOKA_TRAINING_POOL_ID')

JAMENDO_CLIENT_ID = os.getenv('JAMENDO_CLIENT_ID')

GCS_API_KEY = os.getenv('GCS_API_KEY')
GCS_ENGINE_ID = os.getenv('GCS_ENGINE_ID')

STORAGE_URL_STATIC = os.getenv('STORAGE_URL_STATIC')
STORAGE_URL_API = os.getenv('STORAGE_URL_API')

DIR_BASE = os.getcwd() + '/'

# Relative
DIR_DATA = 'data/'
DIR_DATA_AUDIOS = DIR_DATA + 'audios/'
DIR_DATA_LYRICS = DIR_DATA + 'lyrics/'
DIR_DATA_COVERS = DIR_DATA + 'covers/'
DIR_DATA_ARTISTS = DIR_DATA + 'artists/'

FILE_TRACKS = DIR_DATA + 'tracks.csv'
FILE_ARTISTS_INFO = DIR_DATA_ARTISTS + 'artists-info.csv'

FILE_ASSIGNMENTS = DIR_DATA + 'assignments.csv'


def to_absolute_path(relative_path: str) -> str:
    return DIR_BASE + relative_path

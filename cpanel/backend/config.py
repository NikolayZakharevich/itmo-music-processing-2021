from dotenv import load_dotenv
import os

load_dotenv()

BASE_PATH = os.getenv('BASE_PATH')
if not BASE_PATH.endswith('/'):
    BASE_PATH = BASE_PATH + '/'

DATASET_DIR = BASE_PATH + 'data/'
MODELS_DIR = BASE_PATH + 'models/'

DIR_STORAGE_AUDIOS = '/var/www/cpanel/static/data/audios'

IP = os.getenv('IP')
PORT = os.getenv('PORT')

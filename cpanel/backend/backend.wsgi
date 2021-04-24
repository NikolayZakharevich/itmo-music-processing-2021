#!/usr/bin/python
import logging

BASE_PATH = '/var/www/bachelor-thesis-cpanel/'

logging.basicConfig(filename=BASE_PATH + 'logs/log.txt', level=logging.DEBUG)

import sys
sys.path.insert(0, BASE_PATH)

from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
load_dotenv(dotenv_path=Path(BASE_PATH + '.env'), verbose=True)

from app.main import server as application
application.secret_key = 'secret'
import json
import os

import pandas as pd

from flask import Flask
from flask_cors import CORS

from config import *

app = Flask(__name__)
cors = CORS(app)


@app.route("/api/data", methods=['GET'])
def get_data():
    df = pd.read_csv(DATASET_DIR + 'tracks.csv')
    return df.to_json(orient='records'), 200


@app.route("/api/audios_track_ids", methods=['GET'])
def get_audios_track_ids():
    return json.dumps(list(map(
        lambda filename: int(filename.replace('.mp3', '')),
        os.listdir(DIR_STORAGE_AUDIOS)
    ))), 200


def main():
    app.run(host=IP, port=PORT)


if __name__ == "__main__":
    main()

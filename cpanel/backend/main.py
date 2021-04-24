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


def main():
    app.run(host=IP, port=PORT)


if __name__ == "__main__":
    main()

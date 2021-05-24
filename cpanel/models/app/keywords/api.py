import pandas as pd

KEYWORD_WEIGHT_THRESHOLD = 0.5


def kek():
    keywords_df = pd.read_csv(FILE_KEYWORDS)

    track_keywords = {}
    for _, row in keywords_df.iterrows():
        keywords = row.keywords.split(';')
        weights = row.weights.split(';')

        track_keywords[row.track_id] = []
        for i in range(len(keywords)):
            if weights[i] < KEYWORD_WEIGHT_THRESHOLD:
                break
            track_keywords[row.track_id].append(keywords[i])

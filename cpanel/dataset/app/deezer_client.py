import pandas as pd
from deezer import Client, Track, Album, Genre

from app.columns import Column


class Deezer:
    client: Client

    def __init__(self):
        self.client = Client(
            headers={'Accept-Language': 'en'}
        )

    def update_genres(self, tracks: pd.DataFrame) -> pd.DataFrame:
        updated_tracks = []

        for i, row in tracks.iterrows():
            if isinstance(row[Column.DEEZER_GENRES.value], str):
                updated_tracks.append(row)
                continue

            track_title = row[Column.TRACK_TITLE.value]
            artist_name = row[Column.ARTIST_NAME.value]

            try:
                track_search = self.client.search(
                    query=f'{track_title},{artist_name}',
                    relation='track',
                    limit=1
                )
            except Exception as e:
                print(e)
                continue

            if track_search is None or len(track_search) == 0:
                print(f'Failed to get info about track #{i} ({track_title}): no api response')
                continue

            track: Track = track_search[0]
            if not track:
                print(f'Failed to get info about track #{i} ({track_title}): no track in api response')
                continue

            try:
                album: Album = track.get_album()
            except Exception as e:
                print(e)
                continue
            if not album:
                print(f'Failed to get info about track #{i} ({track_title}):: album not found')
                continue

            genres: list[Genre] = album.genres
            genres_str = '|'.join(list(map(lambda g: g.name, genres)))
            print(f'Track #{i}: ym_genre was: {row[Column.YM_GENRE.value]}, deezer_genres found: {genres_str}')

            if genres:
                row[Column.DEEZER_GENRES.value] = genres_str
            updated_tracks.append(row)

        return pd.DataFrame(updated_tracks)

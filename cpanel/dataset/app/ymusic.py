from pathlib import Path
from typing import Any, Union

import pandas as pd
from yandex_music import Client

from app.columns import Emotion, Column, Genre
from app.utils import ensure_dirs_exist
from config import TOKEN_YANDEX_MUSIC, DIR_DATA_AUDIOS, DIR_DATA_LYRICS, DIR_DATA_COVERS, to_absolute_path

PLAYLIST_NAME_PREFIX_EMOTION = 'emotion-'
PLAYLIST_NAME_PREFIX_GENRE = 'genre-'


class YMusic:
    client: Client = None

    CODEC = 'mp3'
    BITRATE = 320
    COVER_SIZE = '1000x1000'

    def update_from_emotion_playlists(
            self,
            tracks: pd.DataFrame,
            emotion: Emotion,
            limit: Union[int, None] = None,
            skip_download=None
    ):
        playlist_name = PLAYLIST_NAME_PREFIX_EMOTION + emotion.value
        return self.update_from_playlist(
            playlist_name=playlist_name,
            current_tracks=tracks,
            limit=limit,
            skip_download=skip_download
        )

    def update_from_genre_playlists(
            self,
            tracks: pd.DataFrame,
            genre: Genre,
            limit: Union[int, None] = None
    ):
        playlist_name = PLAYLIST_NAME_PREFIX_GENRE + genre.value
        return self.update_from_playlist(
            playlist_name=playlist_name,
            current_tracks=tracks,
            limit=limit,
            skip_download=set()
        )

    # @see https://music.yandex.ru/users/nikolay.zakharevich/playlists
    def update_from_playlist(
            self,
            playlist_name: str,
            current_tracks: pd.DataFrame,
            limit: Union[int, None] = None,
            skip_download: Union[None, set[int]] = None
    ):
        self._check_auth()

        emotion_name = None
        if playlist_name.startswith(PLAYLIST_NAME_PREFIX_EMOTION):
            emotion_name = playlist_name[len(PLAYLIST_NAME_PREFIX_EMOTION):]

        updated_tracks: dict[str, dict[str, Any]] = {}
        for i, track in current_tracks.iterrows():
            updated_tracks[str(track[Column.YM_TRACK_ID.value])] = track.to_dict()

        try:
            user_playlists = self.client.users_playlists_list()
        except Exception as e:
            print(e)
            return current_tracks

        playlist = next((p for p in user_playlists if p.title == playlist_name), None)
        if playlist is None:
            print(
                f'Playlist {playlist_name} not found, see https://music.yandex.ru/users/nikolay.zakharevich/playlists')
            return current_tracks

        audio_dir, covers_dir, lyrics_dir = map(
            lambda dir_: Path(to_absolute_path(dir_)), [DIR_DATA_AUDIOS, DIR_DATA_COVERS, DIR_DATA_LYRICS])
        ensure_dirs_exist([audio_dir, covers_dir, lyrics_dir])

        try:
            tracks = playlist.tracks if playlist.tracks else playlist.fetch_tracks()
        except Exception as e:
            print(e)
            return current_tracks

        for i, short_track in enumerate(tracks):
            if isinstance(limit, int) and len(updated_tracks) >= limit:
                break

            try:
                track = short_track.track if short_track.track else short_track.fetchTrack()
            except Exception as e:
                print(e)
                continue
            num = i + 1
            # Section `validation`

            print(f'Processing track #{num} [{playlist_name}]')

            if track is None:
                print(f'Failed to load track #{num}')
                continue

            if track.albums is None or len(track.albums) == 0:
                print(f'Track #{num} has no album')
                continue

            if track.artists is None or len(track.artists) == 0:
                print(f'Invalid track #{num} has no artist')
                continue

            album = track.albums[0]
            artist = track.artists[0]

            if album is None or artist is None:
                print(f'Invalid track #{num}: no album or artist')
                continue

            if album.cover_uri is None:
                print(f'Track #{num} has no cover')
                continue

            has_required_audio_file = any(
                info.codec == self.CODEC and info.bitrate_in_kbps == self.BITRATE for info in track.get_download_info())
            if not has_required_audio_file:
                print(f'Track #{num} ({track.title} has no required audio file to download')
                continue

            # Section `download`

            audio_filename = f'{track.id}.{self.CODEC}'
            cover_filename = f'{track.id}:{self.COVER_SIZE}.jpg'
            lyrics_filename = f'{track.id}.txt'

            if not (audio_dir / Path(audio_filename)).exists():
                if skip_download is None or int(track.id) not in skip_download:
                    try:
                        track.download(audio_dir / Path(audio_filename), codec='mp3', bitrate_in_kbps=320)
                    except Exception as e:
                        print(e)
                        continue
            else:
                print('Audio file already exists')

            if not (covers_dir / Path(cover_filename)).exists():
                try:
                    track.download_cover(covers_dir / Path(cover_filename), size=self.COVER_SIZE)
                except Exception as e:
                    print(e)
                    continue
            else:
                print('Cover file already exists')

            lyrics_path = None
            try:
                supplement = track.get_supplement()
                if supplement is None or supplement.lyrics is None or supplement.lyrics.full_lyrics is None:
                    print(f'Track #{num} ({track.title}) has no lyrics')
                else:
                    if not (lyrics_dir / Path(lyrics_filename)).exists():
                        with open(lyrics_dir / Path(lyrics_filename), "w") as lyrics_file:
                            print(supplement.lyrics.full_lyrics, file=lyrics_file)
                    else:
                        print('Lyrics file already exists')
                    lyrics_path = DIR_DATA_LYRICS + lyrics_filename
            except Exception as e:
                print(e)

            if track.id in updated_tracks:
                print(f'Track #{num} ({track.title}) is already in dataset')

                # Insert new emotion if needed
                if emotion_name is not None:
                    if isinstance(updated_tracks[track.id][Column.EMOTIONS_MAYBE.value], str):
                        current_emotions = updated_tracks[track.id][Column.EMOTIONS_MAYBE.value].split('|')
                        if emotion_name not in current_emotions:
                            current_emotions.append(emotion_name)
                            updated_tracks[track.id][Column.EMOTIONS_MAYBE.value] = '|'.join(current_emotions)
                    else:
                        updated_tracks[track.id][Column.EMOTIONS_MAYBE.value] = emotion_name
            else:
                updated_tracks[track.id] = {
                    Column.YM_TRACK_ID.value: track.id,
                    Column.YM_ALBUM_ID.value: album.id,
                    Column.TRACK_TITLE.value: track.title,
                    Column.ARTIST_NAME.value: artist.name,
                    Column.COVER_URL.value: album.cover_uri.replace('%%', self.COVER_SIZE),
                    Column.YM_GENRE.value: album.genre,
                    Column.YEAR.value: int(album.year) if album.year is not None else None,
                    Column.EMOTIONS_MAYBE.value: emotion_name,

                    Column.AUDIO_PATH.value: DIR_DATA_AUDIOS + audio_filename,
                    Column.LYRICS_PATH.value: lyrics_path,
                    Column.COVER_PATH.value: DIR_DATA_COVERS + cover_filename,

                    Column.ARTIST_COUNTRY.value: None,
                }

        return pd.DataFrame(data=updated_tracks.values(), columns=[col.value for col in Column])

    def _check_auth(self):
        if self.client is None:
            Client.notice_displayed = True
            self.client = Client.from_token(TOKEN_YANDEX_MUSIC, report_new_fields=False)

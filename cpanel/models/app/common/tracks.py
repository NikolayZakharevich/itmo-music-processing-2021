import os

from config import STORAGE_URL_STATIC, DIR_AUDIOS


def get_audio_url(track_id: int) -> str:
    return STORAGE_URL_STATIC + get_audio_path_default(track_id)


def get_audio_path_default(track_id: int) -> str:
    return f'data/audios/{track_id}.mp3'


def audio_file_exists(track_id: int) -> bool:
    return os.path.isfile(get_audio_path_default(track_id))

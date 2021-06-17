import json
import os
import random
from pathlib import Path
from typing import Optional, List, Dict

import cherrypy
import numpy as np
import psutil
import yaml

from app.emotions import predict_topk_emotions, EMOTIONS, get_fonts
from app.features import extract_audio_features
from app.keywords import predict_keywords

METHOD_NAME_EMOTIONS = 'music-emotions'
METHOD_NAME_FONTS = 'music-fonts'
METHOD_NAME_KEYWORDS = 'music-keywords'

ERROR_MESSAGE_NO_FILE_WITH_FEATURES = 'Invalid audio features path `%s`: no such file'
ERROR_MESSAGE_INVALID_FEATURES = 'Failed to extract features from audio (is audio length less than 60 seconds?)'
ERROR_MESSAGE_UNKNOWN_EMOTION = 'Unknown emotion `%s`. Expected emotions: [%s]'
ERROR_MESSAGE_MODEL_FAIL_EMOTIONS = 'Failed to predict emotions: something wrong with model'
ERROR_MESSAGE_MODEL_FAIL_KEYWORDS = 'Failed to predict keywords: something wrong with model'

TypeAudio = cherrypy._cpreqbody.Part

process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open('config.yml'))


class ApiServerController(object):

    @cherrypy.expose(METHOD_NAME_EMOTIONS)
    def music_emotions(self, audio: TypeAudio):
        """
        :param audio: audio file with music song
        """
        features = get_audio_features(audio)
        if len(features) == 0:
            return result_error(ERROR_MESSAGE_INVALID_FEATURES)

        emotions = predict_topk_emotions(np.array(features), k=3)
        if len(emotions) == 0:
            return result_error(ERROR_MESSAGE_MODEL_FAIL_EMOTIONS)

        return result_emotions(emotions)

    @cherrypy.expose(METHOD_NAME_FONTS)
    def music_fonts(self, audio: Optional[TypeAudio] = None, emotion: Optional[str] = None):
        """
        :param audio: audio file with music song
        :param emotion: emotion, selected by user
        """
        if emotion is not None and emotion not in EMOTIONS:
            return result_error(ERROR_MESSAGE_UNKNOWN_EMOTION % (emotion, ', '.join(EMOTIONS)))

        if emotion is not None:
            emotions = [emotion]
        else:
            features = get_audio_features(audio)
            if len(features) == 0:
                return result_error(ERROR_MESSAGE_INVALID_FEATURES)
            emotions = predict_topk_emotions(np.array(features), k=3)

        emotion_fonts = {}
        for emotion in emotions:
            emotion_fonts[emotion] = get_fonts(emotion)
        return result_emotion_fonts(emotion_fonts)

    @cherrypy.expose(METHOD_NAME_KEYWORDS)
    def music_keywords(self, audio: TypeAudio):
        """
        :param audio: audio file with music song
        """
        features = get_audio_features(audio)
        if len(features) == 0:
            return result_error(ERROR_MESSAGE_INVALID_FEATURES)

        keywords = predict_keywords(np.array(features), k=10)
        if len(keywords) == 0:
            return result_error(ERROR_MESSAGE_MODEL_FAIL_KEYWORDS)

        return result_keywords(keywords)


def get_audio_features(audio: TypeAudio) -> List[np.ndarray]:
    """
    :param audio: audio file with music song
    :return: list of features for each full minute
    """
    audio_file_name_prefix = random.randrange(1048576)
    tmp_dir = config['app']['tmp_dir']

    audio_file_path = Path(os.path.join(tmp_dir, f'{audio_file_name_prefix}-{audio.filename}'))
    audio_file_path.parent.mkdir(exist_ok=True, parents=True)
    audio_file_path.write_bytes(audio.file.read())

    features = extract_audio_features(audio_file_path)
    os.remove(audio_file_path)
    return features


def result_error(error_message: str) -> str:
    """
    :param: error_message: error message to return
    """
    return json.dumps({
        'result': {
            'error': error_message
        }
    })


def result_emotions(emotions: List[str]) -> str:
    """
    :param: emotions: list of emotions to return, e.g. ['comfortable', 'happy', 'wary']
    """
    return json.dumps({
        'result': {
            'emotions': emotions
        }
    })


def result_emotion_fonts(emotion_fonts: Dict[str, List[str]]) -> str:
    """
    :param: emotions: fonts grouped by emotion, e.g.
    {
        'comfortable': ['LexendExa', 'Suravaram', 'Philosopher'],
        'happy': ['LilitaOne', 'Acme']
    }
    """
    return json.dumps({
        'result': [
            {'emotion': emotion, 'fonts': fonts} for emotion, fonts in emotion_fonts.items()
        ]
    })


def result_keywords(keywords: List[str]) -> str:
    """
    :param: keywords: list of keywords to return, e.g. ['porn', 'guitar', 'obama']
    """
    return json.dumps({
        'result': {
            'keywords': keywords
        }
    })


if __name__ == '__main__':
    cherrypy.tree.mount(ApiServerController(), '/demo')

    cherrypy.config.update({
        'server.socket_port': config['app']['port'],
        'server.socket_host': config['app']['host'],
        'server.thread_pool': config['app']['thread_pool'],
        'log.access_file': 'access1.log',
        'log.error_file': 'error1.log',
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [('Content-Type', 'text/html;encoding=utf-8')],
    })

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()

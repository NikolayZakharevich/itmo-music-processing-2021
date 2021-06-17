import inspect
import json
import os
import sys
import unittest
from contextlib import contextmanager
from typing import Dict, Optional

import cherrypy
import requests

from urllib.parse import urlencode

# Import trick
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import server

# end of import trick

DIR_DATA = os.path.join('test', 'data')
PATH_AUDIO_1 = os.path.join(DIR_DATA, 'audio-1.mp3')
PATH_AUDIO_2 = os.path.join(DIR_DATA, 'audio-2.mp3')
PATH_AUDIO_3 = os.path.join(DIR_DATA, 'audio-3.mp3')


def data_provider(fn_data_provider):
    """Data provider decorator, allows another callable to provide the data for the test"""

    def test_decorator(fn):
        def repl(self, *args):
            for i in fn_data_provider():
                try:
                    fn(self, *i)
                except AssertionError:
                    print("Assertion error caught with data set ", i)
                    raise

        return repl

    return test_decorator


class EmotionTest(unittest.TestCase):
    audio_emotions = lambda: (
        (PATH_AUDIO_1, ['romantic', 'sweet', 'anger']),
        (PATH_AUDIO_2, ['romantic', 'surprise', 'quiet']),
        (PATH_AUDIO_3, ['romantic', 'sweet', 'anger']),
    )

    method_name = 'music-emotions'

    @data_provider(audio_emotions)
    def test_get_emotions(self, audio_path, expected_emotions):
        cherrypy.tree.mount(server.ApiServerController())
        with run_server(), open(audio_path, 'rb') as audio:
            try:
                response = requests.post(get_url(self.method_name), files={'audio': audio}).json()
                self.assertIsNotNone(response)
                self.assertTrue('result' in response)
                self.assertTrue('emotions' in response['result'])
                self.assertEqual(expected_emotions, response['result']['emotions'])
            except requests.exceptions.RequestException as e:
                print(e)
                self.fail(e)
            except json.decoder.JSONDecodeError as e:
                self.fail('Invalid response format. ' + e.msg)


class FontsTest(unittest.TestCase):
    audio_fonts = lambda: (
        (PATH_AUDIO_1, {
            'romantic': ['Amiri', 'Amarante', 'Fondamento', 'Fondamento', 'Stalemate'],
            'sweet': ['Dekko', 'Kurale', 'Italianno', 'Kenia', 'Rancho'],
            'anger': ['StintUltraCondensed', 'AlfaSlabOne', 'DarkerGrotesque', 'FreckleFace', 'Adriator']
        }),
    )

    emotion_fonts = lambda: (
        ('comfortable', {
            'comfortable': ['LexendExa', 'Suravaram', 'Philosopher'],
        }),
    )

    method_name = 'music-fonts'

    @data_provider(audio_fonts)
    def test_get_audio_fonts(self, audio_path, expected_fonts):
        cherrypy.tree.mount(server.ApiServerController())
        with run_server(), open(audio_path, 'rb') as audio:
            try:
                response = requests.post(get_url(self.method_name), files={'audio': audio}).json()
                self.assertIsNotNone(response)
                self.assertTrue('result' in response)
                for item in response['result']:
                    self.assertTrue('emotion' in item)
                    self.assertTrue('fonts' in item)
                    emotion = item['emotion']
                    self.assertTrue(emotion in expected_fonts)
            except requests.exceptions.RequestException as e:
                print(e)
                self.fail(e)
            except json.decoder.JSONDecodeError as e:
                self.fail('Invalid response format. ' + e.msg)

    @data_provider(emotion_fonts)
    def test_get_emotion_fonts(self, emotion, expected_fonts):
        cherrypy.tree.mount(server.ApiServerController())
        with run_server():
            try:
                response = requests.post(get_url(self.method_name, {'emotion': emotion})).json()
                self.assertIsNotNone(response)
                self.assertTrue('result' in response)
                self.assertTrue('fonts' in response['result'])
            except requests.exceptions.RequestException as e:
                self.fail(e)
            except json.decoder.JSONDecodeError as e:
                self.fail('Invalid response format. ' + e.msg)


class KeywordsTest(unittest.TestCase):
    audio_keywords = lambda: (
        (PATH_AUDIO_1, ['pain', 'soul', 'morning', 'song', 'baby', 'place', 'one', 'heart', 'night', 'dreams']),
    )

    method_name = 'music-keywords'

    @data_provider(audio_keywords)
    def test_get_keywords(self, audio_path, expected_keywords):
        cherrypy.tree.mount(server.ApiServerController())
        with run_server(), open(audio_path, 'rb') as audio:
            try:
                response = requests.post(get_url(self.method_name), files={'audio': audio}).json()
                self.assertIsNotNone(response)
                self.assertTrue('result' in response)
                self.assertTrue('keywords' in response['result'])
                self.assertEqual(expected_keywords, response['result']['keywords'])
            except requests.exceptions.RequestException as e:
                print(e)
                self.fail(e)


@contextmanager
def run_server():
    cherrypy.engine.start()
    cherrypy.engine.wait(cherrypy.engine.states.STARTED)
    yield
    cherrypy.engine.exit()
    cherrypy.engine.block()


def get_url(method_name: str, params: Optional[Dict[str, str]] = None, host='localhost', port=8080) -> str:
    if params is not None:
        params_str = f'?{urlencode(params)}'
    else:
        params_str = ''

    return f'http://{host}:{port}/{method_name}{params_str}'


if __name__ == '__main__':
    unittest.main()

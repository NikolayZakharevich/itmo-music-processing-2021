import os
import random
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urlencode

import cherrypy
import numpy as np
import psutil
import yaml
from cherrypy import log
from mako.lookup import TemplateLookup

from app.emotions import predict_topk_emotions, get_emoji
from app.features import extract_audio_features, save_audio_features, get_dump_path

METHOD_NAME_DEFAULT = 'music-emotions'

DIR_HTML = 'html'

PAGE_STEP_1_2 = 'step1-2.html'

ERROR_MESSAGE_NO_FILE_WITH_FEATURES = 'Invalid audio features path `%s`: no such file'
ERROR_MESSAGE_INVALID_FEATURES = 'Invalid audio features found in file `%s`'

process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open('config.yml'))

method_name = METHOD_NAME_DEFAULT


class AppServerController(object):
    @cherrypy.expose(method_name)
    def music_emotions(
            self,
            step: Optional[int] = None,
            audio: Optional[cherrypy._cpreqbody.Part] = None,
            hash: Optional[str] = None,
    ):
        """
        Dispatching method
        :param step:           - current interactive step
        :param audio:          - audio file with music song
        :param hash:           - part of file name with cached song features
        :param emotion:        - emotion selected by user (among predicted)
        :param emotion_custom: - emotion selected by user (custom)
        :return:
        """
        if isinstance(step, str):
            step = int(step)

        if step is None or step == 1:
            return self.step1_page(audio)
        elif step == 2:
            if hash is None:
                return self.redirect_to(1)
            return self.step2_page(hash)
        else:
            return self.step1_page(audio)

    def step1_page(self, audio: Optional[cherrypy._cpreqbody.Part] = None) -> str:
        """
        Step 1.
        - Initial page
        - Audio uploading

        :param audio: audio file with music song
        :return:
        """
        if audio is not None:
            # Audio uploading
            audio_file_name_prefix = random.randrange(1048576)
            tmp_dir = config['app']['tmp_dir']

            audio_file_path = Path(os.path.join(tmp_dir, f'{audio_file_name_prefix}-{audio.filename}'))
            audio_file_path.parent.mkdir(exist_ok=True, parents=True)
            audio_file_path.write_bytes(audio.file.read())

            features = extract_audio_features(audio_file_path)
            features_dump_path = save_audio_features(features, tmp_dir)
            os.remove(audio_file_path)
            self.redirect_to(2, {'hash': features_dump_path})

        return self.render_step1_page()

    def step2_page(self, hash: str) -> str:
        """
        Step 2
        - Redirect with features file hash
        - Displaying emotion choice

        :param hash:  part of file name with cached song features
        :return:
        """
        features = AppServerController.get_features_by_hash(hash)
        if features is None:
            return self.redirect_to(1)
        emotions = predict_topk_emotions(features, k=3)
        return self.render_step2_page(emotions)

    @staticmethod
    def get_features_by_hash(hash: str) -> Optional[np.ndarray]:
        tmp_dir = config['app']['tmp_dir']
        audio_features_path = get_dump_path(hash, tmp_dir)
        if not os.path.isfile(audio_features_path):
            log(ERROR_MESSAGE_NO_FILE_WITH_FEATURES % audio_features_path)
            return None

        features = np.load(audio_features_path)
        if features is None or len(features) == 0:
            log(ERROR_MESSAGE_INVALID_FEATURES % audio_features_path)
            return None

        return features

    @staticmethod
    def render_step1_page():
        page_params = {}
        return AppServerController.render_page(PAGE_STEP_1_2, **page_params)

    @staticmethod
    def render_step2_page(emotions: List[str]):
        page_params = {}
        for i in range(len(emotions)):
            page_params[f'emotion_{i + 1}'] = emotions[i]
            page_params[f'emoji_{i + 1}'] = get_emoji(emotions[i])
        return AppServerController.render_page(PAGE_STEP_1_2, **page_params)

    @staticmethod
    def render_page(page: str, **kwargs):
        lookup = TemplateLookup(directories=[DIR_HTML])
        tmpl = lookup.get_template(page)
        kwargs['method_name'] = METHOD_NAME_DEFAULT
        return tmpl.render(**kwargs)

    @staticmethod
    def redirect_to(step: int, params: Optional[Dict] = None):
        if params is None:
            params_str = ''
        else:
            params_str = f'&{urlencode(params)}'

        raise cherrypy.HTTPRedirect(f'{method_name}?step={step}{params_str}')


if __name__ == '__main__':
    cherrypy.tree.mount(AppServerController(), '/demo')

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

    if 'method_name' in config['app']:
        method_name = config['app']['method_name']

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()

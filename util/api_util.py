import os
from core.util import basic_io


def get_mapbox_app_token():
    """
    get mapbox api app token as string
    :return: (str) token
    """
    path_to_keys = os.path.join('config', 'mapbox_keys.json')
    keys = basic_io.read_json_to_dict(path_to_keys)
    return keys["app-token"]


def get_socrata_app_token():
    """
    get socrata api app token as string
    :return: (str) token
    """
    path_to_keys = os.path.join('config', 'socrata_chicago_keys.json')
    keys = basic_io.read_json_to_dict(path_to_keys)
    return keys["app_token"]


def get_census_key():
    """
    get census api key as string
    :return: (str) key
    """
    path_to_keys = os.path.join('config', 'census_keys.json')
    keys = basic_io.read_json_to_dict(path_to_keys)
    return keys["api-key"]

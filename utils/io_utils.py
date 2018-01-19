import os
import warnings
from .. import path as gomoku_path


def check_save_path(path):
    try:
        with open(path, 'w') as f:
            return path
    except IOError:
        try:
            with open(gomoku_path +'/'+path, 'w') as f:
                return gomoku_path + '/' + path
        except IOError:
            warnings.warn('the paths:{:s} and {:s} '
                          'don`t exist'.format(path, gomoku_path+ '/' + path))
            return None

def check_load_path(path):
    if os.path.exists(path):
        return path
    elif os.path.exists(gomoku_path+'/'+path):
        return gomoku_path+ '/' + path
    else:
        warnings.warn('the paths:{:s} and {:s} '
                      'don`t exist'.format(path, gomoku_path+ '/' + path))
        return None

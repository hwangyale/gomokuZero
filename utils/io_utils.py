import os
from .. import path as gomoku_path


def check_save_path(path):
    try:
        with open(path, 'w') as f:
            return path
    except IOError:
        with open(gomoku_path +'/'+path, 'w') as f:
            return gomoku_path + '/' + path

def check_load_path(path):
    if os.path.exists(path):
        return path
    elif os.path.exists(gomoku_path+'/'+path):
        return gomoku_path+ '/' + path
    else:
        raise Exception('the paths:{:s} and {:s} '
                        'don`t exist'.format(path, gomoku_path+ '/' + path))

from __future__ import print_function

import os
from .. import path as gomoku_path


def check_save_path(path, warning_flag=True):
    if os.path.exists(path):
        return path
    try:
        with open(path, 'w') as f:
            pass
        os.remove(path)
        return path
    except IOError:
        path_suffix = path
        path = gomoku_path + '/' + path_suffix
        if os.path.exists(path):
            return path
        try:
            with open(path, 'w') as f:
                pass
            os.remove(path)
            return path
        except IOError:
            if warning_flag:
                print('the paths:{:s} and {:s} '
                      'don`t exist'.format(path_suffix, path))
            return None

def check_load_path(path, warning_flag=True):
    if os.path.exists(path):
        return path
    elif os.path.exists(gomoku_path+'/'+path):
        return gomoku_path+ '/' + path
    else:
        if warning_flag:
            print('the paths:{:s} and {:s} '
                  'don`t exist'.format(path, gomoku_path+ '/' + path))
        return None

def get_cache_path(folder_path, suffix):
    count = int(id(folder_path + suffix))
    while True:
        file_path = folder_path + str(count) + suffix
        if check_load_path(file_path, False) is not None:
            count += 1
            continue
        if check_save_path(file_path, False) is not None:
            return check_save_path(file_path, False)
        else:
            print('please check the folder path: {:s} and '
                  'the suffix of the file: {:s}'.format(folder_path, suffix))

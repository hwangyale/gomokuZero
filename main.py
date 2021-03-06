from __future__ import print_function

import os
import Tkinter
import time

from gomokuZero.train.pipeline import Trainer
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.utils.io_utils import check_load_path
from gomokuZero.utils import thread_utils

try:
    input = raw_input
except NameError:
    pass

class Controller(thread_utils.Thread):
    def __init__(self, stopFlag, lock):
        self.stopFlag = stopFlag
        self.lock = lock
        super(Controller, self).__init__()

    def run(self):
        lock = self.lock
        root = Tkinter.Tk()
        label = Tkinter.Label(root)
        label['text'] = 'Stop Training?'
        button = Tkinter.Button(root)
        button['text'] = 'yes'

        def on_click():
            lock.acquire()
            if self.stopFlag[0]:
                self.stopFlag[0] = False
                label['text'] = 'Stop Training?'
            else:
                self.stopFlag[0] = True
                label['text'] = 'Continue?'
            lock.release()

        button['command'] = on_click
        label.pack()
        button.pack()

        root.mainloop()

def run(save_file_path, save_json_path,
        save_weights_path, save_data_path,
        cache_file_path, cache_json_path,
        cache_weights_path, cache_data_path):
    if check_load_path(cache_file_path) is not None:
        trainer = Trainer.load(cache_file_path)

    else:
        default = {
            'blocks': 2,
            'kernel_size': (3, 3),
            'filters': 64,
            'game_number': 50,
            'step_to_explore': 6,
            'game_batch_size': 25,
            'augment': True,
            'pool_size': 250,
            'pool_point': 0,
            'rollout_time': 250,
            'max_thread': 2,
            'epochs': 1000,
            'train_batch_size': 256,
            'train_epochs': 4,
            'lr': {
                400: 1e-2,
                600: 1e-3,
                float('inf'): 1e-4
            },
            'momentum': 0.9,
            'value_loss_weight': 1.0,
            'weight_decay': 1e-4
        }
        if check_load_path(save_json_path) is not None:
            pvn = PolicyValueNetwork.load_model(check_load_path(save_json_path))
        else:
            pvn = None
        trainer = Trainer(pvn, **default)

    stopFlag = [False]
    lock = thread_utils.lock
    controller = Controller(stopFlag, lock)
    controller.daemon = True
    controller.start()
    trainer.run(
        save_file_path=save_file_path,
        save_json_path=save_json_path,
        save_weights_path=save_weights_path,
        save_data_path=save_data_path,
        cache_file_path=cache_file_path,
        cache_json_path=cache_json_path,
        cache_weights_path=cache_weights_path,
        cache_data_path=cache_data_path,
        stopFlag=stopFlag
    )


if __name__ == '__main__':
    # version = 'test_version_'
    # save_prefix = 'data/zero/'

    version = 'pre_train_version_'
    save_prefix = 'data/pre_train/'

    cache_prefix = 'data/cache/cache_'

    run(
        save_file_path=save_prefix + version + 'setting.json',
        save_json_path=save_prefix + version + 'nn_config.json',
        save_weights_path=save_prefix + version + 'nn_weights.h5',
        save_data_path=save_prefix + version + 'samples.npz',
        cache_file_path=cache_prefix + version + 'setting.json',
        cache_json_path=cache_prefix + version + 'nn_config.json',
        cache_weights_path=cache_prefix + version + 'nn_weights.h5',
        cache_data_path=cache_prefix + version + 'samples.npz'
    )

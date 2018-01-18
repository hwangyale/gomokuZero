from __future__ import print_function

import os
import threading
import Tkinter
import time

from gomokuZero.train.pipeline import Trainer
from gomokuZero.model.neural_network import PolicyValueNetwork

try:
    input = raw_input
except NameError:
    pass

class Controller(threading.Thread):
    def __init__(self, stopFlag, lock):
        self.stopFlag = stopFlag
        self.lock = lock
        super(Controller, self).__init__()

    def run(self):
        lock = self.lock
        root = Tkinter.Tk()
        button = Tkinter.Button(root)
        button['text'] = 'Stop Training?'

        def on_click():
            lock.acquire()
            self.stopFlag[0] = True
            lock.release()
            button['text'] = 'Saving...'

        button['command'] = on_click
        button.pack()

        root.mainloop()

def run(save_file_path, save_json_path,
        save_weights_path, save_data_path,
        cache_file_path, cache_json_path,
        cache_weights_path, cache_data_path):
    if os.path.exists(cache_file_path):
        trainer = Trainer.load(cache_file_path)

    else:
        default = {
            'blocks': 3,
            'kernel_size': (3, 3),
            'filters': 64,
            'game_number': 128,
            'step_to_explore': 8,
            'game_batch_size': 8,
            'augment': True,
            'rollout_time': 512,
            'max_thread': 8,
            'epochs': 256,
            'train_steps': 1024,
            'train_batch_size': 128,
            'lr': {
                400: 1e-2,
                600: 1e-3,
                float('inf'): 1e-4
            },
            'momentum': 0.9,
            'value_loss_weight': 1.0,
            'weight_decay': 1e-4
        }
        trainer = Trainer(**default)

    stopFlag = [False]
    lock = threading.RLock()
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
    save_prefix = 'data/zero/'
    cache_prefix = 'data/cache/cache_'
    version = 'test_version_'
    run(
        save_file_path=save_prefix + version + 'setting.json',
        save_json_path=save_prefix + version + 'nn_config.json',
        save_weights_path=save_prefix + version + 'nn_weights.npz',
        save_data_path=save_prefix + version + 'samples.npz',
        cache_file_path=cache_prefix + version + 'setting.json',
        cache_json_path=cache_prefix + version + 'nn_config.json',
        cache_weights_path=cache_prefix + version + 'nn_weights.npz',
        cache_data_path=cache_prefix + version + 'samples.npz'
    )

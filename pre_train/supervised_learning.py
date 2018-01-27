from __future__ import print_function

import sys
import gc
import json
import numpy as np
import random
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from ..constant import *
from ..board.board import Board
from ..utils.preprocess_utils import roting_fliping_functions, augment_data
from ..utils.io_utils import check_load_path, check_save_path
from ..utils.progress_bar import ProgressBar
from ..model.neural_network import PolicyValueNetwork
from ..model.preprocess import Preprocessor
from ..train.optimizers import StochasticGradientDescent, Adam, OptimizerSaving

try:
    range = xrange
except NameError:
    pass

def get_samples_from_history(history_pool, augment=True, save_path=None, shuffle=True):
    preprocessor = Preprocessor()
    size = sum([len(history) for history in history_pool])
    board_tensors = []
    policy_tensors = []
    value_tensors = []
    progress_bar = ProgressBar(len(history_pool))
    count = 0
    for idx, history in enumerate(history_pool, 1):
        board = Board()
        samples = []
        for position in history:
            if len(position) == 3 and position[-1] == 0:
                board.move(position[:2])
                continue
            board_tensor = preprocessor.get_inputs(board)
            policy_tensor = np.zeros((SIZE, SIZE), dtype=np.float32)
            policy_tensor[position[:2]] = 1.0
            policy_tensor = np.expand_dims(policy_tensor, axis=0)
            player = board.player
            if not augment and shuffle:
                func = random.choice(roting_fliping_functions)
                board_tensor = func(board_tensor)
                policy_tensor = func(policy_tensor)
            if len(position) == 3:
                value_weight = 1.0 if position[-1] < 1000 else VALUE_WEIGHT
            else:
                value_weight = 1.0
            samples.append((board_tensor, policy_tensor, player, value_weight))

            board.move(position[:2])

        winner = board.winner
        weight_flag = False
        for sample in samples[::-1]:
            board_tensor, policy_tensor, player, value_weight = sample

            if not weight_flag and player == winner and value_weight == VALUE_WEIGHT:
                weight_flag = True
            if weight_flag:
                value_weight = VALUE_WEIGHT
            else:
                value_weight = 1.0

            if winner == DRAW:
                value = 0.0
                value_weight = VALUE_WEIGHT
            elif winner == player:
                value = 1.0
            else:
                value = -1.0

            board_tensors.append(board_tensor)
            policy_tensors.append(policy_tensor)
            value_tensors.append(np.array([value, value_weight]).reshape((1, 2)))

        progress_bar.update(idx)

    board_tensors = np.concatenate(board_tensors, axis=0)
    policy_tensors = np.concatenate(policy_tensors, axis=0)
    value_tensors = np.concatenate(value_tensors, axis=0)

    if save_path is not None:
        sys.stdout.write(' '*79 + '\r')
        sys.stdout.flush()
        sys.stdout.write('saving...')
        sys.stdout.flush()
        np.savez(
            check_save_path(save_path),
            board_tensors=board_tensors,
            policy_tensors=policy_tensors,
            value_tensors=value_tensors
        )
    sys.stdout.write(' '*79 + '\r')
    sys.stdout.flush()
    gc.collect()
    return board_tensors, policy_tensors, value_tensors


class Trainer(object):
    def __init__(self, sample_path, json_path,
                 weights_path, optimizer_path,
                 batch_size, epochs,
                 save_path,
                 history_path='data/records/yixin_records.npz',
                 current_epoch=0,
                 train_idxs=None,
                 **nn_config):
        self.paths = {
            'sample_path': sample_path,
            'json_path': json_path,
            'weights_path': weights_path,
            'optimizer_path': optimizer_path,
            'history_path': history_path,
            'save_path': save_path
        }
        self.batch_size = batch_size
        self.epochs = epochs
        self.nn_config = nn_config

        self.current_epoch = current_epoch
        if train_idxs is not None:
            self.train_idxs = train_idxs

    def get_samples(self):
        if hasattr(self, 'board_tensors'):
            return self.board_tensors, self.policy_tensors, self.value_tensors

        sample_path = self.paths['sample_path']
        if check_load_path(sample_path) is None:
            print('get samples from history pool')
            history_path = self.paths['history_path']
            history_pool = np.load(check_load_path(history_path))['history_pool']
            board_tensors, policy_tensors, value_tensors = get_samples_from_history(
                history_pool, False, check_save_path(sample_path), True
            )
        else:
            samples = np.load(check_load_path(sample_path))
            board_tensors = samples['board_tensors']
            policy_tensors = samples['policy_tensors']
            value_tensors = samples['value_tensors']

        self.board_tensors = board_tensors
        self.policy_tensors = policy_tensors
        self.value_tensors = value_tensors

        return board_tensors, policy_tensors, value_tensors

    def get_pvn(self):
        if hasattr(self, 'pvn'):
            return self.pvn

        json_path = self.paths['json_path']
        if check_load_path(json_path) is None:
            print('initialize the weights of `pvn`')
            pvn = PolicyValueNetwork(**self.nn_config)
            pvn.save_model(
                json_path, self.paths['weights_path']
            )
        else:
            pvn = PolicyValueNetwork.load_model(check_load_path(json_path))

        self.pvn = pvn
        return pvn

    def get_optimizer(self):
        if hasattr(self, 'optimizer'):
            return self.optimizer

        optimizer_path = self.paths['optimizer_path']
        if check_load_path(optimizer_path) is None:
            optimizer = StochasticGradientDescent(lr=0.1, momentum=0.9, nesterov=True)
            # optimizer = Adam(lr=1e-3)
        else:
            with open(check_load_path(optimizer_path), 'r') as f:
                config = json.load(f)
            optimizer = StochasticGradientDescent.from_config(config)
            # optimizer = Adam.from_config(config)

        self.optimizer = optimizer
        return optimizer

    def get_callbacks(self):
        if hasattr(self, 'callbacks'):
            return self.callbacks

        weights_path = self.paths['weights_path']
        modelSaving = ModelCheckpoint(
            check_save_path(weights_path), save_weights_only=True
        )

        def scheduler(epoch):
            if epoch <= 60:
                return 0.05
            if epoch <= 120:
                return 0.01
            if epoch <= 160:
                return 0.01
            return 0.0004

        lrChanging = LearningRateScheduler(scheduler)

        def on_epoch_end(epoch, logs):
            self.save_trainer(epoch)

        optimizerSaving = OptimizerSaving(
            self.get_optimizer(),
            self.paths['optimizer_path']
        )

        configSaving = LambdaCallback(on_epoch_end=on_epoch_end)

        callbacks = [modelSaving, lrChanging, optimizerSaving, configSaving]
        # callbacks = [modelSaving, optimizerSaving, configSaving]
        self.callbacks = callbacks
        return callbacks

    def fit(self):
        pvn = self.get_pvn()

        optimizer = self.get_optimizer()
        pvn.model.compile(
            optimizer=optimizer,
            loss={'p': 'categorical_crossentropy',
                  'v': mean_weighted_squared_error},
            loss_weights={'p': 1.0, 'v': 1.0}
        )

        board_tensors, policy_tensors, value_tensors = self.get_samples()
        size = board_tensors.shape[0]
        idxs = list(range(size))
        if hasattr(self, 'train_idxs'):
            train_idxs = self.train_idxs
            test_idxs = list(set(idxs)-set(train_idxs))
        else:
            random.shuffle(idxs)
            split = int(np.ceil(size*0.1))
            train_idxs = idxs[split:]
            test_idxs = idxs[:split]
            self.train_idxs = train_idxs

        board_train = board_tensors[train_idxs, ...]
        policy_train = policy_tensors[train_idxs, ...]
        value_train = value_tensors[train_idxs, ...]

        board_train, policy_train, value_train = augment_data(
            board_train, policy_train, value_train
        )
        policy_train = np.reshape(policy_train, (-1, SIZE**2))

        if len(test_idxs):
            board_test = board_tensors[test_idxs, ...]
            policy_test = policy_tensors[test_idxs, ...]
            value_test = value_tensors[test_idxs, ...]

            board_test, policy_test, value_test = augment_data(
                board_test, policy_test, value_test
            )
            policy_test = np.reshape(policy_test, (-1, SIZE**2))

            validation_data = (board_test, [policy_test, value_test])
        else:
            validation_data = None

        callbacks = self.get_callbacks()

        self.save_trainer(self.current_epoch-1)

        pvn.model.fit(
            board_train,
            [policy_train, value_train],
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=self.current_epoch,
            callbacks=callbacks,
            validation_data=validation_data
        )

    def save_trainer(self, epoch):
        self.current_epoch = epoch + 1

        config = {
            'paths': self.paths,
            'nn_config': self.nn_config,
            'current_epoch': self.current_epoch,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'train_idxs': self.train_idxs if hasattr(self, 'train_idxs') else None
        }

        save_path = self.paths['save_path']
        with open(check_save_path(save_path), 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_trainer(cls, save_path):
        with open(check_load_path(save_path), 'r') as f:
            config = json.load(f)

        kwargs = {
            'current_epoch': config['current_epoch'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'train_idxs': config.get('train_idxs', None)
        }
        kwargs.update(config['paths'])
        kwargs.update(config['nn_config'])

        return cls(**kwargs)


def mean_weighted_squared_error(y_true, y_pred):
    return K.mean(y_true[:, 1:] * K.square(y_pred - y_true[:, :1]), axis=-1)

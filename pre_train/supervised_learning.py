from __future__ import print_function

import sys
import gc
import json
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from ..constant import *
from ..board.board import Board
from ..utils.preprocess_utils import roting_fliping_functions
from ..utils.io_utils import check_load_path, check_save_path
from ..utils.progress_bar import ProgressBar
from ..model.neural_network import PolicyValueNetwork
from ..model.preprocess import Preprocessor
from ..train.optimizers import StochasticGradientDescent, OptimizerSaving

def get_samples_from_history(history_pool, augment=True, save_path=None):
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
            board_tensor = preprocessor.get_inputs(board)
            policy_tensor = np.zeros((SIZE, SIZE), dtype=np.float32)
            policy_tensor[position] = 1.0
            policy_tensor = np.expand_dims(policy_tensor, axis=0)
            player = board.player
            samples.append((board_tensor, policy_tensor, player))

            board.move(position)

        winner = board.winner
        for sample in samples:
            board_tensor, policy_tensor, player = sample
            if winner == DRAW:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0

            board_tensors.append(board_tensor)
            policy_tensors.append(policy_tensor)
            value_tensors.append(np.array(value).reshape((1, 1)))

        progress_bar.update(idx)

    board_tensors = np.concatenate(board_tensors, axis=0)
    policy_tensors = np.concatenate(policy_tensors, axis=0)
    value_tensors = np.concatenate(value_tensors, axis=0)

    if augment:
        augment_board_tensors = []
        augment_policy_tensors = []
        augment_value_tensors = []
        for idx, func in enumerate(roting_fliping_functions):
            sys.stdout.write(' '*79 + '\r')
            sys.stdout.flush()
            sys.stdout.write('function index:{:d}\r'.format(idx))
            sys.stdout.flush()
            augment_board_tensors.append(func(board_tensors))
            augment_policy_tensors.append(func(policy_tensors))
            augment_value_tensors.append(value_tensors)
        board_tensors = np.concatenate(augment_board_tensors, axis=0)
        policy_tensors = np.concatenate(augment_policy_tensors, axis=0)
        value_tensors = np.concatenate(augment_value_tensors, axis=0)

    policy_tensors = np.reshape(policy_tensors, (-1, SIZE**2))
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

    def get_samples(self):
        if hasattr(self, 'board_tensors'):
            return self.board_tensors, self.policy_tensors, self.value_tensors

        sample_path = self.paths['sample_path']
        if check_load_path(sample_path) is None:
            print('get samples from history pool')
            history_path = self.paths['history_path']
            history_pool = np.load(check_load_path(history_path))['history_pool']
            board_tensors, policy_tensors, value_tensors = get_samples_from_history(
                history_pool, True, check_save_path(sample_path)
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
        else:
            with open(check_load_path(optimizer_path), 'r') as f:
                config = json.load(f)
            optimizer = StochasticGradientDescent.from_config(config)

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
        self.callbacks = callbacks
        return callbacks

    def fit(self):
        pvn = self.get_pvn()

        optimizer = self.get_optimizer()
        pvn.model.compile(
            optimizer=optimizer,
            loss={'p': 'categorical_crossentropy',
                  'v': 'mean_squared_error'},
            loss_weights={'p': 1.0, 'v': 1e-2}
        )

        board_tensors, policy_tensors, value_tensors = self.get_samples()
        callbacks = self.get_callbacks()

        pvn.model.fit(
            board_tensors,
            [policy_tensors, value_tensors],
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=self.current_epoch,
            callbacks=callbacks
        )

    def save_trainer(self, epoch):
        self.current_epoch = epoch + 1

        config = {
            'paths': self.paths,
            'nn_config': self.nn_config,
            'current_epoch': self.current_epoch,
            'batch_size': self.batch_size,
            'epochs': self.epochs
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
            'epochs': config['epochs']
        }
        kwargs.update(config['paths'])
        kwargs.update(config['nn_config'])

        return cls(**kwargs)

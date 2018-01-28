from __future__ import print_function

import sys
import os
import gc
import numpy as np
import random
import json
import collections
from keras.callbacks import LearningRateScheduler
from ..constant import *
from ..board.board import Board
from ..model.neural_network import PolicyValueNetwork
from ..model.mcts import MCTS
from ..model.preprocess import Preprocessor
from .optimizers import StochasticGradientDescent
from ..utils import tolist, check_load_path, check_save_path
from ..utils.progress_bar import ProgressBar
from ..utils.preprocess_utils import roting_fliping_functions, augment_data


try:
    range = xrange
except NameError:
    pass


def get_samples_by_self_play(
        pvn, game_number, step_to_explore,
        game_batch_size=32,
        **kwargs
    ):
    mcts = MCTS(pvn.copy(), **kwargs)
    preprocessor = Preprocessor()
    game_count = game_number
    game_over = 0
    boards = set()
    epsilon = kwargs.get('exploration_epsilon', 0.25)

    samples = collections.defaultdict(list)

    def collect_samples(_boards):
        for board in _boards:
            if board.is_over:
                winner = board.winner
            else:
                continue
            for sample in samples[board]:
                player = sample[-1]
                if winner == DRAW:
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0
                sample[-1] = value

    progress_bar = ProgressBar(game_number)
    while boards or game_count:
        if boards:
            average_steps = int(np.mean([len(board.history) for board in boards]))
            sys.stdout.write(' '*79 + '\r')
            sys.stdout.flush()
            sys.stdout.write(' '*70 + 'ave:{}\r'.format(average_steps))
            sys.stdout.flush()
        progress_bar.update(game_over)

        for _ in range(min(game_batch_size-len(boards), game_count)):
            boards.add(Board())
            game_count -= 1
        cache_boards = list(boards)
        exploration_epsilon = [0.0 if len(board.history) < step_to_explore else epsilon
                               for board in cache_boards]
        taus = [1.0 if len(board.history) < step_to_explore else 0.0
                for board in cache_boards]
        policies = mcts.get_policies(cache_boards, Tau=taus,
                                     exploration_epsilon=exploration_epsilon)
        policies = tolist(policies)
        for board, policy in zip(cache_boards, policies):
            board_tensor = preprocessor.get_inputs(board)
            player = board.player
            policy_tensor = np.zeros((SIZE, SIZE))
            for l_p, prob in policy.iteritems():
                policy_tensor[l_p] = prob
            policy_tensor = np.expand_dims(policy_tensor, axis=0)
            samples[board].append([board_tensor, policy_tensor, player])

        positions = mcts.get_positions(cache_boards, Tau=taus,
                                       exploration_epsilon=exploration_epsilon)
        positions = tolist(positions)

        finished_boards = []
        for board, position in zip(cache_boards, positions):
            board.move(position)
            if board.is_over:
                finished_boards.append(board)

        for board in finished_boards:
            boards.remove(board)
            game_over += 1

        if len(finished_boards):
            collect_samples(finished_boards)
            mcts.clear(finished_boards)

    sys.stdout.write(' '*79 + '\r')
    sys.stdout.flush()
    progress_bar.update(game_over)

    return samples.values()

def get_samples_from_history(history_pool):
    preprocessor = Preprocessor()
    samples = []
    for history in history_pool:
        board = Board()
        samples.append([])
        for position in history:
            if len(position) == 3 and position[-1] == 0:
                board.move(position[:2])
                continue
            board_tensor = preprocessor.get_inputs(board)
            player = board.player
            policy_tensor = np.zeros((SIZE, SIZE))
            policy_tensor[position[:2]] = 1.0
            policy_tensor = np.expand_dims(policy_tensor, axis=0)
            samples[-1].append([board_tensor, policy_tensor, player])

            board.move(position[:2])

        winner = board.winner
        for sample in samples[-1]:
            player = sample[-1]
            if winner == DRAW:
                value = 0.0
            elif player == winner:
                value = 1.0
            else:
                value = -1.0
            sample[-1] = value

    return samples

def process_samples(samples, augment=False):
    board_tensors = []
    policy_tensors = []
    value_tensors = []
    for _samples in samples:
        for sample in _samples:
            board_tensors.append(sample[0])
            policy_tensors.append(sample[1])
            value_tensors.append(np.array(sample[2]).reshape((1, 1)))
    board_tensors = np.concatenate(board_tensors, axis=0)
    policy_tensors = np.concatenate(policy_tensors, axis=0)
    value_tensors = np.concatenate(value_tensors, axis=0)

    if augment:
        augment_board_tensors = []
        augment_policy_tensors = []
        augment_value_tensors = []
        for func in roting_fliping_functions:
            augment_board_tensors.append(func(board_tensors))
            augment_policy_tensors.append(func(policy_tensors))
            augment_value_tensors.append(value_tensors)

        board_tensors = np.concatenate(augment_board_tensors, axis=0)
        policy_tensors = np.concatenate(augment_policy_tensors, axis=0)
        value_tensors = np.concatenate(augment_value_tensors, axis=0)

    return board_tensors, policy_tensors.reshape((-1, SIZE**2)), value_tensors


class Trainer(object):
    '''the trainer for training the model

    #arguments
        init_pvn

        blocks
        kernel_size
        filters

        game_number
        step_to_explore
        game_batch_size
        pool_size
        pool_point
        augment

        rollout_time
        max_thread

        epochs
        train_batch_size
        train_epochs

        lr
        momentum

        value_loss_weight
        weight_decay
    '''
    def __init__(self, init_pvn=None, **kwargs):
        default = {
            'blocks': 3,
            'kernel_size': (3, 3),
            'filters': 16,
            'game_number': 128,
            'step_to_explore': 3,
            'game_batch_size': 32,
            'pool_size': 2000,
            'pool_point': 0,
            'augment': True,
            'rollout_time': 512,
            'max_thread': 64,
            'epochs': 4,
            'train_batch_size': 1024,
            'train_epochs': 3,
            'lr': {
                400: 1e-2,
                600: 1e-3,
                float('inf'): 1e-4
            },
            'momentum': 0.9,
            'value_loss_weight': 1.0,
            'weight_decay': 1e-4
        }
        assert len(set(kwargs.keys()) - set(default.keys())) == 0
        default.update(kwargs)
        self.__dict__.update(default)
        self.setting = default

        if init_pvn is None:
            setting = {
                'blocks': self.blocks,
                'kernel_size': self.kernel_size,
                'filters': self.filters,
                'weight_decay': self.weight_decay
            }
            self.pvn = PolicyValueNetwork(**setting)
        else:
            self.pvn = init_pvn

    def get_scheduler(self):
        pairs = sorted(self.lr.items(), key=lambda (e, l): e)
        def scheduler(epoch):
            for pair in pairs:
                if epoch < pair[0]:
                    return pair[1]
            return pairs[-1][1]
        return LearningRateScheduler(scheduler)

    def run(self,
            save_file_path, save_json_path,
            save_weights_path, save_data_path,
            cache_file_path=None, cache_json_path=None,
            cache_weights_path=None, cache_data_path=None,
            stopFlag=None,
            history_pool='data/records/yixin_records.npz'):
        if cache_file_path is None:
            cache_file_path = save_file_path
        if cache_json_path is None:
            cache_json_path = save_json_path
        if cache_weights_path is None:
            cache_weights_path = save_weights_path
        if cache_data_path is None:
            cache_data_path = save_data_path

        if hasattr(self, 'current_epoch'):
            current_epoch = self.current_epoch
        else:
            current_epoch = 0

        pvn = self.pvn

        pool_size = self.pool_size
        pool_point = self.pool_point
        sample_pool = [None for _ in range(pool_size)]

        if hasattr(self, 'samples'):
            samples = self.samples
        elif check_load_path(cache_data_path) is not None:
            samples = np.load(check_load_path(cache_data_path))['samples']
        elif check_load_path(history_pool) is not None:
            history_pool = np.load(check_load_path(history_pool))['history_pool']
            random.shuffle(history_pool)
            samples = get_samples_from_history(history_pool[:pool_size])
        else:
            samples = []

        for idx in range(min(len(samples), pool_size)):
            sample_pool[pool_point % pool_size] = samples[idx]
            pool_point += 1

        for epoch in range(current_epoch, self.epochs):
            if stopFlag is not None and stopFlag[0]:
                break
            print('epoch:{:d}/{:d}'.format(epoch+1, self.epochs))
            print('self-play:')
            samples = get_samples_by_self_play(
                pvn=pvn,
                game_number=self.game_number,
                step_to_explore=self.step_to_explore,
                game_batch_size=self.game_batch_size,
                rollout_time=self.rollout_time,
                max_thread=self.max_thread
            )
            sys.stdout.write(' '*79 + '\r')
            sys.stdout.flush()
            print('get {:d} samples'.format(len(samples)))
            print('optimization:')

            random.shuffle(samples)
            for idx in range(min(len(samples), pool_size)):
                sample_pool[pool_point % pool_size] = samples[idx]
                pool_point += 1

            self.samples = sample_pool[:pool_point]
            self.pool_point = pool_point

            board_tensors, policy_tensors, value_tensors = process_samples(
                self.samples, augment=self.augment
            )

            if K.image_data_format() == 'channels_last':
                board_tensors = np.transpose(board_tensors, (0, 2, 3, 1))

            optimizer = StochasticGradientDescent(momentum=self.momentum)
            pvn.model.compile(
                optimizer=optimizer,
                loss={'p': 'categorical_crossentropy',
                      'v': 'mean_squared_error'},
                loss_weights={'p': 1.0, 'v': self.value_loss_weight}
            )
            pvn.model.fit(
                x=board_tensors,
                y=[policy_tensors, value_tensors],
                batch_size=self.train_batch_size,
                epochs=self.train_epochs,
                verbose=1,
                callbacks=[self.get_scheduler()]
            )

            self.current_epoch = epoch + 1

            self.save(cache_file_path, cache_json_path,
                      cache_weights_path, cache_data_path)
            print('\n')
            gc.collect()

        self.save(save_file_path, save_json_path,
                  save_weights_path, save_data_path)

    def save(self, file_path, json_path, weights_path, data_path):
        setting = self.setting
        config = {
            'setting': setting,
            'json_path': json_path,
            'data_path': data_path,
            'current_epoch': self.current_epoch,
        }
        self.pvn.save_model(json_path, weights_path)
        np.savez(check_save_path(data_path), samples=self.samples)
        with open(file_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(cls, file_path):
        with open(check_load_path(file_path), 'r') as f:
            config = json.load(f)
        pvn = PolicyValueNetwork.load_model(config['json_path'])
        trainer = cls(pvn, **config['setting'])
        trainer.current_epoch = config['current_epoch']
        if check_load_path(config['data_path']) is not None:
            data = np.load(check_load_path(config['data_path']))
            trainer.samples = data['samples']
        return trainer

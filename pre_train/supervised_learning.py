from __future__ import print_function

import sys
import gc
import numpy as np
from ..constant import *
from ..board.board import Board
from ..utils.preprocess_utils import roting_fliping_functions
from ..utils.io_utils import check_save_path
from ..utils.progress_bar import ProgressBar
from ..model.preprocess import Preprocessor

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

    return board_tensor, policy_tensors, value_tensors

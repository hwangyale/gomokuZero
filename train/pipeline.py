import sys
import numpy as np
from ..constant import *
from ..board.board import Board
from ..model.neural_network import PolicyValueNetwork
from ..model.mcts import MCTS
from ..model.preprocess import Preprocessor
from .optimizers import StochasticGradientDescent
from ..utils import tolist
from ..utils.progress_bar import ProgressBar


try:
    range = xrange
except NameError:
    pass


def get_samples(pvn, game_number, step_to_explore, batch_size=32, **kwargs):
    mcts = MCTS(pvn.copy(), **kwargs)
    preprocessor = Preprocessor()
    game_count = game_number
    game_over = 0
    boards = set()
    samples = []
    epsilon = kwargs.get('exploration_epsilon', 0.25)
    pb = ProgressBar(game_number)
    while boards or game_count:
        pb.update(game_over)
        for _ in range(min(batch_size-len(boards), game_count)):
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
            samples.append((board_tensor, policy_tensor, player, board))

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

    board_tensors = []
    policy_tensors = []
    value_tensors = []
    for sample in samples:
        board_tensor, policy_tensor, player, board = sample
        if board.winner == DRAW:
            value = 0.0
        elif board.winner == player:
            value = 1.0
        else:
            value = -1.0
        board_tensors.append(board_tensor)
        policy_tensors.append(policy_tensor)
        value_tensors.append(value)

    board_tensors = np.concatenate(board_tensors, axis=0)
    policy_tensors = np.concatenate(policy_tensors, axis=0)
    value_tensors = np.array(value_tensors).reshape((-1, 1))

    return board_tensors, policy_tensors, value_tensors


class Trainer(object):
    def __init__(self, )

import sys
from ..board.board import Board
from ..model.neural_network import PolicyValueNetwork
from ..model.mcts import MCTS
from .optimizers import StochasticGradientDescent
from ..utils import tolist


try:
    range = xrange
except NameError:
    pass


def get_samples(pvn, game_number, step_to_explore, batch_size=32, **kwargs):
    player = MCTS(pvn.copy(), **kwargs)
    game_count = game_number
    boards = {}
    samples = []
    epsilon = kwargs.get('exploration_epsilon', 0.25)
    while boards or game_count:
        for _ in range(min(batch_size-len(boards), game_count)):
            boards.add(Board())
            game_count -= 1
        cache_boards = list(boards)
        exploration_epsilon = [0.0 if len(board.history) < step_to_explore else epsilon
                               for board in cache_boards]
        taus = [1.0 if len(board.history) < step_to_explore else 0.0
                for board in cache_boards]
        policies = player.get_policies(cache_boards, Tau=taus,
                                       exploration_epsilon=exploration_epsilon)
        policies = tolist(policies)
        
        positions = player.get_positions(cache_boards, Tau=taus,
                                         exploration_epsilon=exploration_epsilon)
        positions = tolist(positions)

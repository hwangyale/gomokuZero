from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS
from gomokuZero.board.board import Board

pvn = PolicyValueNetwork(blocks=3, filters=16)
mcts = MCTS(pvn)
board = Board()
mcts.rollout(board, 1.0, 16, 4, 4)

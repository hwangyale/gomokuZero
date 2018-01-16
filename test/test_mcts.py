from __future__ import print_function
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS
from gomokuZero.board.board import Board

pvn = PolicyValueNetwork(blocks=3, filters=16)
mcts = MCTS(pvn)
print(mcts.get_policies(Board(), 1.0, 1000, 8))
print(mcts.get_positions(Board(), 1.0, 1000, 8))

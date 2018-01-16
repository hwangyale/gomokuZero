from __future__ import print_function

from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.board.board import Board
from gomokuZero.board.play import PlayerBase, Game

class NeuralNetworkPlayer(PlayerBase):
    def __init__(self, neuralNetwork):
        self.neuralNetwork = neuralNetwork

    def get_position(self, board):
        return self.neuralNetwork.get_positions(board, True)

# player_1 = NeuralNetworkPlayer(PolicyValueNetwork())
# player_2 = NeuralNetworkPlayer(PolicyValueNetwork())
# game = Game(player_1, player_2, time_delay=1)
# game.play()
pvn = PolicyValueNetwork()
board = Board([(8, 8)])
print(pvn.get_policy_values([board, board], True))
print(pvn.get_position_values([board, board], True))

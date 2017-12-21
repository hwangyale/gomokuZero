from __future__ import print_function

from abc import ABCMeta, abstractmethod
import os
import random
import warnings
from ..common import *
from .board import Board


try:
    range = xrange
except NameError:
    pass


class PlayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_action(self, board, **kwargs):
        pass


class Player(PlayerBase):
    def get_action(self, board, **kwargs):
        while True:
            inputs = raw_input('where to place the piece '
                               '(inputs seperated by space):\n').split()
            if len(inputs) != 2:
                warnings.warn('illegal inputs')
                continue
            try:
                row = int(inputs[0])
                col = int(inputs[1])
                action = (row-1, col-1)
            except ValueError:
                warnings.warn('illegal inputs')
                continue
            if not check_border(action):
                warnings.warn('cross the border')
                continue
            if action not in board.legal_actions:
                warnings.warn('illegal action')
                continue
            return action


class Game(object):
    def __init__(self, black_player, white_player):
        self.black_player = black_player
        self.white_player = white_player

    def play(self, board=None, **kwargs):
        if board is None:
            board = Board()
        while not board.is_over:
            os.system('cls')
            print(board)
            player = {BLACK: self.black_player,
                      WHITE: self.white_player}[board.player]
            action = player.get_action(board, **kwargs.get(player, {}))
            board.move(action)
        os.system('cls')
        print(board)
        result = '\n'
        if board.winner == DRAW:
            result += 'draw'
        elif board.winner == BLACK:
            result += 'black wins'
        else:
            result += 'white wins'
        print(result)

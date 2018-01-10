from __future__ import print_function

from abc import ABCMeta, abstractmethod
import os
import random
from ..common import *
from .board import Board


try:
    range = xrange
except NameError:
    pass


action2position = lambda (r, c): (r-1, c-1)

position2action = lambda (r, c): (r+1, c+1)


class PlayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_position(self, board, **kwargs):
        pass


class Player(PlayerBase):
    def get_position(self, board, **kwargs):
        while True:
            inputs = raw_input('where to place the piece '
                               '(inputs seperated by space):\n').split()
            if len(inputs) != 2:
                print('illegal inputs\n')
                continue
            try:
                row = int(inputs[0])
                col = int(inputs[1])
                position = action2position((row, col))
            except ValueError:
                print('illegal inputs\n')
                continue
            if not check_border(position):
                print('cross the border\n')
                continue
            if position not in board.legal_positions:
                print('illegal action\n')
                continue
            return position


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
            position = player.get_position(board, **kwargs.get(player, {}))
            board.move(position)
        os.system('cls')
        print(board)
        result = '\n'
        if board.winner == DRAW:
            result += 'draw'
        else:
            if board.winner == BLACK:
                result += 'black wins'
            else:
                result += 'white wins'

            start = position2action(board.five[0])
            end = position2action(board.five[1])
            result += ' from {} to {}'.format(start, end)
        print(result)

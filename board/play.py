from __future__ import print_function
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import sys
import os
import random
import time
import numpy as np
from ..constant import *
from ..utils.board_utils import *
from ..utils.progress_bar import ProgressBar
from ..utils import tolist
from .board import Board


try:
    range = xrange
except NameError:
    pass

try:
    input = raw_input
except NameError:
    pass


action2position = lambda r_c: (r_c[0]-1, r_c[1]-1)

position2action = lambda r_c: (r_c[0]+1, r_c[1]+1)


class PlayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_position(self, board, **kwargs):
        pass


class Player(PlayerBase):
    def get_position(self, board, **kwargs):
        while True:
            inputs = input('where to place the piece '
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

class AIPlayer(PlayerBase):
    def __init__(self, ai):
        self.ai = ai

    def get_position(self, board, **kwargs):
        return self.ai.get_positions(board, **kwargs)

class Game(object):
    def __init__(self, black_player, white_player, **kwargs):
        self.black_player = black_player
        self.white_player = white_player
        self.time_delay = kwargs.get('time_delay', 0)

    def switch(self):
        self.black_player, self.white_player = self.white_player, self.black_player

    def play(self, board=None, **kwargs):
        if board is None:
            board = Board()
        time_delay = self.time_delay
        os.system('cls')
        print(board)
        print('\n')
        while not board.is_over:
            player = {BLACK: self.black_player,
                      WHITE: self.white_player}[board.player]
            position = player.get_position(board, **kwargs)
            board.move(position)
            time.sleep(time_delay)
            os.system('cls')
            print(board)
            print('move: ({:d}, {:d})'.format(position[0]+1, position[1]+1))
            print('\n')
        os.system('cls')
        print(board)
        print('\n')
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


class Tournament(object):
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2

    def play(self, game_number, batch_size, **kwargs):
        player_1 = self.player_1
        player_2 = self.player_2
        progress_bar = ProgressBar(2*game_number)
        player_1.win = 0.0
        player_2.win = 0.0
        step = 0
        for black, white in [(player_1, player_2), (player_2, player_1)]:
            current, opponent = black, white
            boards = set()
            count = 0
            while count < game_number:
                if current == black:
                    while len(boards) < min(batch_size, game_number-count):
                        boards.add(Board())
                cache_boards = list(boards)
                positions = tolist(current.get_positions(cache_boards, **kwargs))
                finished_boards = []
                for board, position in zip(cache_boards, positions):
                    board.move(position)
                    if board.is_over:
                        if board.winner != DRAW:
                            {BLACK: black, WHITE: white}[board.winner].win += 1.0
                        finished_boards.append(board)

                average_step = np.mean([len(board.history) for board in boards])
                sys.stdout.write(' '*79 + '\r')
                sys.stdout.flush()
                sys.stdout.write(' '*60 + 'mean step:{:.0f}\r'.format(average_step))
                sys.stdout.flush()

                for board in finished_boards:
                    boards.remove(board)
                    count += 1
                    step += 1

                progress_bar.update(step)

                current, opponent = opponent, current

        sys.stdout.write(' '*79 + '\r')
        sys.stdout.flush()
        print('player 1 win:{:.2f}%'.format(player_1.win/2/game_number*100))
        print('player 2 win:{:.2f}%'.format(player_2.win/2/game_number*100))

from __future__ import print_function

import os
import time
import random
import threading
from gomokuZero.board.board import Board
from gomokuZero.utils.board_utils import *
from gomokuZero.utils.gomoku_utils import get_urgent_positions
from gomokuZero.utils.vct import get_vct
from gomokuZero.utils.thread_utils import lock
from gomokuZero.board.play import PlayerBase


class VCTPlayer(PlayerBase):
    def __init__(self, max_depth=225, max_time=100):
        self.max_depth = max_depth
        self.max_time = max_time
        self.lock = lock
        self.container = dict()

    def get_position(self, board):
        start = time.time()
        value, positions = get_vct(board, self.max_depth, self.max_time)
        end = time.time()
        print('vct searching time:{:.4f}'.format(end - start))
        if not value:
            raise Exception('There does not exist vct')
        return positions[0]


class DefensePlayer(PlayerBase):
    def get_position(self, board):
        positions = get_urgent_positions(board)
        positions = list(positions)
        if len(positions):
            return random.choice(positions)
        else:
            raise Exception('There does not exist vct')


def play_based_on_vct_record(history, max_depth=225, max_time=100, time_delay=2):
    board = Board(history)
    current = VCTPlayer(max_depth, max_time)
    opponent = DefensePlayer()
    while not board.is_over:
        os.system('cls')
        print(board)
        position = current.get_position(board)
        board.move(position)
        time.sleep(time_delay)
        current, opponent = opponent, current
    os.system('cls')
    print(board)


# record = [
#     [(7, 7), (8, 7), (9, 7), (8, 6), (7, 6), (7, 5),
#      (6, 6), (7, 8), (6, 4)]
# ]
#
# record = [
#     [(7, 7), (7, 6), (7, 8), (8, 6), (6, 8), (8, 8),
#      (6, 6), (6, 5), (6, 7)]
# ]

record = [
    (7, 7), (6, 8), (8, 6), (6, 7), (6, 6), (7, 8),
    (5, 7), (5, 6), (4, 5), (5, 5), (7, 5), (4, 8),
    (5, 8), (9, 3), (8, 4), (7, 3)
]


play_based_on_vct_record(record)

from __future__ import print_function

import os
import time
import random
import threading
from gomokuZero.board.board import Board
from gomokuZero.utils.board_utils import *
from gomokuZero.utils.gomoku_utils import get_urgent_positions
from gomokuZero.utils.vct import get_vct, Node
from gomokuZero.utils.thread_utils import lock
from gomokuZero.board.play import PlayerBase


try:
    input = raw_input
except NameError:
    pass


class VCTPlayer(PlayerBase):
    def __init__(self, max_depth=225, max_time=100, global_threat=True, included_four=True):
        self.max_depth = max_depth
        self.max_time = max_time
        self.lock = lock
        self.global_threat = global_threat
        self.included_four = included_four
        self.container = dict()

    def get_position(self, board):
        start = time.time()
        value, positions = get_vct(board, self.max_depth, self.max_time,
                                   global_threat=self.global_threat, included_four=self.included_four)
        end = time.time()
        print('vct searching time:{:.4f}'.format(end - start))
        if not value:
            raise Exception('There does not exist vct')
        return positions[0]


class DefensePlayer(PlayerBase):
    def get_position(self, board):
        current_positions, opponent_positions = get_promising_positions(board)
        # print(current_positions)
        # print(opponent_positions)
        positions = get_urgent_positions(board)
        positions = list(positions)
        if len(positions):
            # return random.choice(positions)
            while True:
                position = input('option:{:s}\n'.format(str([(p[0]+1, p[1]+1) for p in positions])))
                try:
                    r, c = position.split()
                    r = int(r) - 1
                    c = int(c) - 1
                except:
                    pass
                if (r, c) in positions:
                    return (r, c)
        else:
            raise Exception('There does not exist vct')


def play_based_on_vct_record(history, max_depth=225, max_time=100, time_delay=2,
                             global_threat=True, included_four=True):
    board = Board(history)
    current = VCTPlayer(max_depth, max_time, global_threat=global_threat,
                        included_four=included_four)
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
#     (7, 7), (8, 7), (9, 7), (8, 6), (7, 6), (7, 5),
#     (6, 6), (7, 8), (6, 4)
# ]

# record = [
#     (7, 7), (7, 6), (7, 8), (8, 6), (6, 8), (8, 8),
#     (6, 6), (6, 5), (6, 7)
# ]

# record = [
#     (7, 7), (6, 8), (8, 6), (6, 7), (6, 6), (7, 8),
#     (5, 7), (5, 6), (4, 5), (5, 5), (7, 5), (4, 8),
#     (5, 8), (9, 3), (8, 4), (7, 3)
# ]

# record = [
#     (7, 7), (6, 7), (6, 8), (7, 8), (6, 6), (6, 5),
#     (5, 6), (4, 6), (5, 5), (4, 4), (7, 6), (5, 9),
#     (8, 6), (9, 6), (8, 7), (8, 5)
# ]

# record = [
#     (7, 6), (8, 6), (8, 5), (7, 5), (9, 4), (10, 3),
#     (9, 7), (9, 6), (10, 5), (6, 7), (11, 6), (12, 7),
#     (11, 5), (10, 6), (12, 5), (9, 5)
# ]

# record = [
#     (7, 7), (8, 7), (8, 8), (7, 8), (9, 6), (9, 7),
#     (6, 6), (5, 5), (8, 5), (7, 4), (8, 6), (7, 6),
#     (9, 5), (10, 4), (8, 4), (7, 5), (7, 3), (10, 6),
#     (8, 3), (8, 2), (11, 5), (10, 5), (10, 7), (11, 8),
#     (10, 2), (11, 10), (6, 5)
# ]

record = [
    (7, 7), (6, 6), (6, 7), (8, 7), (7, 8), (5, 7),
    (4, 8), (7, 9), (6, 8), (5, 8), (5, 9), (4, 10),
    (6, 10), (7, 11), (8, 9), (9, 10), (9, 8), (7, 10),
    (9, 7)
]


play_based_on_vct_record(record, max_depth=20, max_time=3600, global_threat=True, included_four=True)
# play_based_on_vct_record(record, max_depth=225, max_time=3600)
print('created {:d} nodes during searching'.format(Node.count))

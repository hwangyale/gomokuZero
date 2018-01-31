from __future__ import print_function

from gomokuZero.utils.board_utils import *
from gomokuZero.board.board import Board

try:
    input = raw_input
except NameError:
    pass

history = [(8, 7), (9, 6), (11, 6), (11, 7), (7, 7), (6, 7),
           (9, 8), (8, 8), (7, 6), (10, 9), (8, 3), (7, 4)]

board = Board()
for r, c in history:
    print(board)
    print(get_promising_positions(board)[0])
    print(get_promising_positions(board)[1])
    print('\n')
    board.move((r, c))

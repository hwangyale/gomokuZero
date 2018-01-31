from __future__ import print_function

from gomokuZero.utils.board_utils import *
from gomokuZero.board.board import Board

try:
    input = raw_input
except NameError:
    pass

board = Board()
while True:
    try:
        r, c = input('').split()
        r = int(r)
        c = int(c)
    except:
        exit()
    board.move((r, c))
    print(board)
    print(get_promising_positions(board))
    print('\n')

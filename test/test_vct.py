from __future__ import print_function

import threading
from gomokuZero.board.board import Board
from gomokuZero.utils.vct import VCT

# board = Board(
#     [(7, 7), (8, 7), (9, 7), (8, 6), (7, 6), (7, 5),
#      (6, 6), (7, 8), (6, 4)]
# )
board = Board(
    [(7, 7), (7, 6), (7, 8), (8, 6), (6, 8), (8, 8),
     (6, 6), (6, 5), (6, 7)]
)
print(board)
container = dict()
vct = VCT(board, container, threading.RLock(), 225, 100)
vct.start()
vct.join()
print(container[board])

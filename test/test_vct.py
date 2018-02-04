from __future__ import print_function

import time
import threading
from gomokuZero.board.board import Board
from gomokuZero.utils.vct import VCT

# board = Board(
#     [(7, 7), (8, 7), (9, 7), (8, 6), (7, 6), (7, 5),
#      (6, 6), (7, 8), (6, 4)]
# )
# board = Board(
#     [(7, 7), (7, 6), (7, 8), (8, 6), (6, 8), (8, 8),
#      (6, 6), (6, 5), (6, 7)]
# )
board = Board(
    [(7, 7), (6, 8), (8, 6), (6, 7), (6, 6), (7, 8),
     (5, 7), (5, 6), (4, 5), (5, 5), (7, 5), (4, 8),
     (5, 8), (9, 3), (8, 4), (7, 3)]
)
print(board)
container = dict()
vct = VCT(board, container, threading.RLock(), 225, 100)
start = time.time()
vct.start()
vct.join()
end = time.time()
print(container[board])
print('time:{:.4f}'.format(end - start))

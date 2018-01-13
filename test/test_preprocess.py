import numpy as np
from gomokuZero.constant import *
from gomokuZero.board.board import Board
from gomokuZero.utils.preprocess_utils import roting_fliping_functions
from gomokuZero.model.preprocess import Preprocessor

# a = np.arange(32).reshape((2, 4, 4))
# print rot090(a)
# print rot180(a)
# print rot270(a)
# print rot360(a)
# print flip_row(a)
# print flip_col(a)
# print flip_lef(a)
# print flip_rig(a)

preprocessor = Preprocessor()
board = Board([(8, 8), (9, 9)])
print preprocessor.get_inputs(board, roting_fliping_functions[0]).shape
print preprocessor.get_outputs(np.random.rand(1, SIZE, SIZE), board).shape

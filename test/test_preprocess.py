import numpy as np
from gomokuZero.constant import *
from gomokuZero.board.board import Board
from gomokuZero.utils.preprocess_utils import roting_fliping_functions
from gomokuZero.model.preprocess import Preprocessor

preprocessor = Preprocessor()
board = Board([(8, 8), (9, 9)])
print(preprocessor.get_inputs(board, roting_fliping_functions[0]).shape)
print(preprocessor.get_outputs(np.random.rand(1, SIZE, SIZE), board).shape)

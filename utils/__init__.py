from .io_utils import *
from .nn_utils import NeuralNetworkBase, NeuralNetworkDecorate
from .board_utils import *
from .preprocess_utils import *

def tolist(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

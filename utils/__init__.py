import numpy as np
from .io_utils import *
from .nn_utils import NeuralNetworkBase, NeuralNetworkDecorate
from .board_utils import *
from .preprocess_utils import *

def tolist(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def sample(ps):
    '''the sum of `ps` must be equal to 1
    '''
    if isinstance(ps, list):
        s = 0.0
        random_value = np.random.rand()
        for idx, p in enumerate(ps):
            s += p
            if random_value < s:
                return idx
        return len(ps) - 1

    elif isinstance(ps, dict):
        keys = ps.keys()
        values = ps.values()
        s = 0.0
        random_value = np.random.rand()
        for idx, p in enumerate(values):
            s += p
            if random_value < s:
                return values[idx]
        return values[-1]

    else:
        raise Exception('Unknown type of `ps`: {}'.format(type(ps)))

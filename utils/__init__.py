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
    if isinstance(ps, dict):
        keys = ps.keys()
        values = ps.values()
        s = 0.0
        random_value = np.random.rand()
        for idx, p in enumerate(values):
            s += p
            if random_value < s:
                return keys[idx]
        return keys[-1]

    s = 0.0
    random_value = np.random.rand()
    for idx, p in enumerate(ps):
        s += p
        if random_value < s:
            return idx
    return len(ps) - 1

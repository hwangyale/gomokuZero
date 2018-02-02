import numpy as np
from keras import backend as K
from keras.engine.training import _make_batches
from ..constant import *
from .preprocess_utils import roting_fliping_functions


class AugmentationGenerator(object):
    def __init__(self, board_tensors, policy_tensors, value_tensors):
        self.board_tensors = board_tensors
        self.policy_tensors = policy_tensors
        self.value_tensors = value_tensors

    def get_generator(self, batch_size):
        board_tensors = self.board_tensors
        policy_tensors = self.policy_tensors
        value_tensors = self.value_tensors
        size = board_tensors.shape[0] * 8
        func_number = len(roting_fliping_functions)
        batches = _make_batches(size, batch_size)
        dimshuffle = True if K.image_data_format() == 'channels_last' else False
        def generator():
            indice = list(range(size))
            while True:
                np.random.shuffle(indice)
                for batch in batches:
                    idxs = indice[batch[0]:batch[1]]
                    cache_board_tensors = []
                    cache_policy_tensors = []
                    cache_value_tensors = []
                    for idx in idxs:
                        sample_idx = idx // func_number
                        func = roting_fliping_functions[idx % func_number]
                        cache_board_tensors.append(func(board_tensors[sample_idx:sample_idx+1, ...]))
                        cache_policy_tensors.append(func(policy_tensors[sample_idx:sample_idx+1, ...]))
                        cache_value_tensors.append(value_tensors[sample_idx:sample_idx+1, ...])

                    cache_board_tensors = np.concatenate(cache_board_tensors, axis=0)
                    cache_policy_tensors = np.concatenate(cache_policy_tensors, axis=0).reshape((-1, SIZE**2))
                    cache_value_tensors = np.concatenate(cache_value_tensors, axis=0)

                    if dimshuffle:
                        cache_board_tensors = np.transpose(cache_board_tensors, (0, 2, 3, 1))

                    yield (cache_board_tensors, [cache_policy_tensors, cache_value_tensors])

        return generator(), len(batches)

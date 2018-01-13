from ..constant import *
from ..utils.preprocess_utils import *
from ..utils import tolist


class Preprocessor(object):
    def __init__(self):
        self.steps = HISTORY_STEPS
        self.boards2inverseFuncs = {}

    @property
    def shape(self):
        return (None, self.steps*2, SIZE, SIZE)

    def _get_input(self, board, func=None):
        steps = self.steps
        history = board.history
        player = board.player
        opponent = {BLACK: WHITE, WHITE: BLACK}[player]
        board = np.array(board._board)
        tensor = np.zeros((1, )+self.shape[1:], dtype=np.float32)
        for step in range(min(steps, len(history))):
            tensor[0, step, ...] = (board == player)
            tensor[0, step+steps, ...] = (board == opponent)
            board[history[-step-1]] = EMPTY

        if func is None:
            return tensor
        else:
            self.boards2inverseFuncs[board] = inverse_mapping[func]
            return func(tensor)

    def get_inputs(self, boards, func=None):
        boards = tolist(boards)
        if func is None or callable(func):
            funcs = [None] * len(boards)
        elif isinstance(func, list):
            if len(boards) != len(func):
                raise Exception('{:d} `boards` can`t match {:d} `func`'
                                .format(len(boards), len(func)))
            funcs = func
        else:
            raise Exception('Unknown `func`')

        tensors = []
        for board, func in zip(boards, funcs):
            tensors.append(self._get_input(board, func))
        tensors = np.concatenate(tensors, axis=0)
        if callable(func):
            inverse_func = inverse_mapping[func]
            self.boards2inverseFuncs.update(
                {board: inverse_func for board in boards}
            )
            return func(tensors)
        else:
            return tensors

    def get_outputs(self, distributions, boards, inverse_func=None):
        shape = distributions.shape
        if len(shape) in [1, 2]:
            distributions = distributions.reshape((-1, SIZE, SIZE))
            shape = distributions.shape
        elif len(shape) != 3:
            raise Exception('Unknown shape of distributions: {:s}'
                            .format(str(shape)))

        n = shape[0]
        boards = tolist(boards)
        if len(boards) != n:
            raise Exception('{:d} `distributions` can`t match {:d} `boards`'
                            .format(n, len(boards)))

        if inverse_func is None:
            inverse_funcs = []
            for board in boards:
                inverse_funcs.append(self.boards2inverseFuncs.get(board, None))
            if len(set(inverse_funcs)) == 1:
                if inverse_funcs[0] is None:
                    return distributions
                else:
                    return inverse_funcs[0](distributions)

        elif isinstance(inverse_func, list):
            if len(inverse_func) != n:
                raise Exception('{:d} `boards` can`t match {:d} `inverse_func`'
                                .format(n, len(inverse_func)))
            inverse_funcs = inverse_func

        elif callable(inverse_func):
            return inverse_func(distributions)

        for idx, func in enumerate(inverse_funcs):
            if func is not None:
                distributions[idx, ...] = func(distributions[idx, ...])

        return distributions

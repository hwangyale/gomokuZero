from . import tolist
from ..constant import *
from memory_profiler import profile


def rollout_function(boards, pvn, vct_max_depth=4, vct_max_time=0.01):
    boards = tolist(boards)
    players = [board.player for board in boards]
    idxs2boards = {idx: board for idx, board in enumerate(boards) if not board.is_over}
    while len(idxs2boards):
        idxs = list(idxs2boards.keys())
        unfinished_boards = list(idxs2boards.values())
        positions = tolist(pvn.get_positions(unfinished_boards,
                                             vct_max_depth=vct_max_depth,
                                             vct_max_time=vct_max_time))
        for idx, unfinished_board, position in zip(idxs, unfinished_boards, positions):
            unfinished_board.move(position)
            if unfinished_board.is_over:
                del idxs2boards[idx]

    values = []
    for player, board in zip(players, boards):
        winner = board.winner
        if winner == DRAW:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        values.append(value)

    if len(values) == 1:
        return values[0]
    else:
        return values

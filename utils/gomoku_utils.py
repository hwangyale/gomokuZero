__all__ = ['get_urgent_positions', 'get_neighbours']

import copy
import collections
from ..constant import *
from .board_utils import *


def get_container_from_original_board(board, container_name):
    if hasattr(board, container_name):
        return board.__dict__[container_name]
    if board.original_board is not None \
            and hasattr(board.original_board, container_name):
        original_container = board.original_board.__dict__[container_name]
        if isinstance(original_container, collections.defaultdict):
            container = collections.defaultdict(original_container.default_factory)
            for key, value in original_container.iteritems():
                container[key] = copy.copy(value)
        elif isinstance(original_container, dict):
            container = dict()
            for key, value in original_container.iteritems():
                container[key] = copy.copy(value)
        else:
            container = copy.copy(original_container)

        board.__dict__[container_name] = container

        return container

    return None


def get_urgent_positions(board):
    current_positions, opponent_positions = get_promising_positions(board)
    if len(board.history) < 6:
        return []
    if len(current_positions[OPEN_FOUR]):
        return current_positions[OPEN_FOUR]
    elif len(opponent_positions[OPEN_FOUR]) or len(opponent_positions[FOUR]):
        return opponent_positions[OPEN_FOUR] | opponent_positions[FOUR]
    elif len(current_positions[OPEN_THREE]):
        return current_positions[OPEN_THREE]
    elif len(opponent_positions[OPEN_THREE]):
        return current_positions[THREE] | opponent_positions[OPEN_THREE]
    return []


NEIGHBOURS = [
    [
        [
            f((r, c), sign*delta)
            for sign in [-1, 1]
            for delta in range(1, 3)
            for f in move_list
            if check_border(f((r, c), sign*delta))
        ]
        for c in range(SIZE)
    ]
    for r in range(SIZE)
]
def get_neighbours(board):
    if len(board.history) >= (SIZE-2)**2:
        return board.legal_positions

    neighbours = get_container_from_original_board(board, 'neighbours')
    if neighbours is None:
        neighbours = set()
        board.neighbours = neighbours

    if len(board.history):
        if not hasattr(board, '_neighbours_update_step'):
            board._neighbours_update_step = 0
        if board._neighbours_update_step == len(board.history)-1:
            r, c = board.history[-1]
            neighbours.discard((r, c))
            for position in NEIGHBOURS[r][c]:
                if position in board.legal_positions:
                    neighbours.add(position)

            board._neighbours_update_step += 1

        elif board._neighbours_update_step < len(board.history)-1:
            cache_neighbours = set()
            for r, c in board.history[board._neighbours_update_step:]:
                neighbours.discard((r, c))
                for position in NEIGHBOURS[r][c]:
                    cache_neighbours.add(position)
            cache_neighbours &= board.legal_positions
            neighbours |= cache_neighbours
            board.neighbours = neighbours

            board._neighbours_update_step += 1

    if len(neighbours):
        return neighbours
    else:
        return {(r, c) for r in range(SIZE) for c in range(SIZE)}

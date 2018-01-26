__all__ = ['get_threats', 'get_neighbours']

import copy
import collections
from ..constant import *
from .board_utils import check_border, move_list

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


def _get_threats(board):
    if len(board.history) < 6:
        return None

    _board = board._board

    def check(position, color, to_block):
        positions = set()
        for m_f in move_list:
            counts = {-1: 0, 0: 1, 1: 0}
            cache_positions = [None] * 4
            for sign in [-1, 1]:
                empty_flag = 0
                for delta in range(1, 5):
                    r, c = m_f(position, sign*delta)
                    if check_border((r, c)):
                        if _board[r][c] == color:
                            if empty_flag == 0:
                                counts[0] += 1
                                continue
                            elif empty_flag == 1:
                                counts[sign] += 1
                                continue

                        elif _board[r][c] == EMPTY and empty_flag < 2:
                            empty_flag += 1
                            cache_positions[sign+empty_flag] = (r, c)
                            continue

                    break

            if sum(counts.values()) < 3 \
                    or cache_positions[0] is None \
                    or cache_positions[2] is None:
                continue

            for sign in [-1, 1]:
                if counts[0] + counts[sign] >= 3:
                    if cache_positions[sign+2] is not None:
                        positions.add(cache_positions[sign+1])
                        if to_block and counts[0] < 3:
                            positions.add(cache_positions[sign+2])
                            positions.add(cache_positions[-sign+1])

        return positions

    all_positions = set()

    player = board.player
    opponent = {BLACK: WHITE, WHITE: BLACK}[player]

    threats = get_container_from_original_board(board, 'threats')
    if threats is None:
        threats = collections.defaultdict(set)
        board.threats = threats

    for color, to_block_flag, idx in [(player, False, -2), (opponent, True, -1)]:
        position_set = threats[color]
        position_set.add(board.history[idx])
        remove_positions = []
        for position in position_set:
            cache = check(position, color, to_block_flag)
            if cache:
                all_positions |= cache
            else:
                remove_positions.append(position)

        for position in remove_positions:
            position_set.remove(position)

        if all_positions:
            return list(all_positions)

if NUMBER == 5:
    get_threats = _get_threats
else:
    def get_threats(*args, **kwargs):
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
            if (r, c) in neighbours:
                neighbours.remove((r, c))
            for position in NEIGHBOURS[r][c]:
                if position in board.legal_positions:
                    neighbours.add(position)

            board._neighbours_update_step += 1

        elif board._neighbours_update_step < len(board.history)-1:
            cache_neighbours = set()
            for r, c in board.history[board._neighbours_update_step:]:
                if (r, c) in neighbours:
                    neighbours.remove((r, c))
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
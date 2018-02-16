import gc
import os
import sys
import json
import numpy as np
import random
import collections
from ..constant import *
from .. import path as gomokuZero_path


try:
    range = xrange
    VERSION = 2
except NameError:
    VERSION = 3


def check_border(action):
    return 0 <= action[0] < SIZE and 0 <= action[1] < SIZE

def move_function(action, delta):
    return (action[0]+delta[0], action[1]+delta[1])

hor_move = lambda action, d=1: move_function(action, (0, d))
ver_move = lambda action, d=1: move_function(action, (d, 0))
dia_move = lambda action, d=1: move_function(action, (d, d))
bac_move = lambda action, d=1: move_function(action, (-d, d))

move_list = [hor_move, ver_move, dia_move, bac_move]
move_dict = {'hor': hor_move, 'ver': ver_move,
             'dia': dia_move, 'bac': bac_move}


#hashing
hashing_keys_file = gomokuZero_path + '\\utils\\hashing_keys_of_size_{:d}.json'.format(SIZE)
if os.path.exists(hashing_keys_file):
    with open(hashing_keys_file, 'r') as f:
        HASHING_KEYS = json.load(f)
    for r, c in [(x, y) for x in range(SIZE) for y in range(SIZE)]:
        for color in [BLACK, WHITE]:
            HASHING_KEYS[r][c][color] = HASHING_KEYS[r][c].pop(str(color))
else:
    HASHING_KEYS = [
        [
            {color: random.randint(0, 2**(SIZE**2)) for color in [BLACK, WHITE]}
            for _ in range(SIZE)
        ]
        for _ in range(SIZE)
    ]
    with open(hashing_keys_file, 'w') as f:
        json.dump(HASHING_KEYS, f)

color_mapping = {BLACK: WHITE, WHITE: BLACK}

def get_hashing_key_of_board(board):
    if hasattr(board, 'zobrist_key'):
        zobrist_key = board.zobrist_key
        zobrist_step = board.zobrist_step
        color = [BLACK, WHITE][zobrist_step % 2]
    elif hasattr(board, 'original_board') \
            and hasattr(board.original_board, 'zobrist_key'):
        zobrist_key = board.original_board.zobrist_key
        zobrist_step = board.original_board.zobrist_step
    else:
        zobrist_key = 0
        zobrist_step = 0
    color = [BLACK, WHITE][zobrist_step % 2]

    for r, c in board.history[zobrist_step:]:
        zobrist_key ^= HASHING_KEYS[r][c][color]
        color = color_mapping[color]

    board.zobrist_key = zobrist_key
    board.zobrist_step = len(board.history)
    return zobrist_key


#get gomoku types
OPEN_FOUR = 1
FOUR = 2
OPEN_THREE = 3
THREE = 4
OPEN_TWO = 5
TWO = 6

GOMOKU_TYPES = [OPEN_FOUR, FOUR, OPEN_THREE, THREE, OPEN_TWO, TWO]

#improve searching by using hashing
HASHING_TABLE_OF_INDICES = dict()


def get_promising_positions(board):
    if hasattr(board, '_promising_positions'):
        if board._searched_step == len(board.history):
            current_positions, opponent_positions = board._promising_positions
            return current_positions, opponent_positions
        else:
            base_gomoku_types = board._gomoku_types
            searched_step = board._searched_step

    else:
        promising_positions = None
        searched_step = 0
        base_gomoku_types = {
            color: {gt: set() for gt in GOMOKU_TYPES}
            for color in [BLACK, WHITE]
        }
        _board = board
        while _board.original_board is not None:
            _board = _board.original_board
            if hasattr(_board, '_promising_positions'):
                promising_positions = _board._promising_positions
                searched_step = _board._searched_step
                base_gomoku_types = _board._gomoku_types
                break
        if promising_positions is not None and searched_step == len(board.history):
            current_positions, opponent_positions = promising_positions
            return current_positions, opponent_positions

    (current_positions, opponent_positions), gomoku_types = _get_promising_positions(
        np.array(board._board).tolist(),
        base_gomoku_types,
        board.history[searched_step:],
        board.player
    )
    board._promising_positions = (current_positions, opponent_positions)
    board._gomoku_types = gomoku_types
    board._searched_step = len(board.history)

    return current_positions, opponent_positions

def _get_promising_positions(board, base_gomoku_types, history, player):
    opponent = color_mapping[player]
    gomoku_types = dict()
    history = history[::-1]
    for color_for_searching in [BLACK, WHITE]:
        start = (color_for_searching == player) + 0
        additional_positions = set(history[start::2])
        _gomoku_types = {
            t: s | additional_positions
            for t, s in base_gomoku_types[color_for_searching].items()
        }
        gomoku_types[color_for_searching] = _gomoku_types

    def check(_position, _color, _move_function, max_empty_count, max_delta=6):
        _cache_counts = {i*j: 0 for i in [-1, 1] for j in range(1, max_empty_count)}
        _cache_counts[0] = 1
        _cache_positions = {i*j: None for i in [-1, 1] for j in range(1, max_empty_count)}
        for sign in [-1, 1]:
            empty_count = 0
            for delta in range(1, max_delta):
                r, c = _move_function(_position, sign*delta)
                if not check_border((r, c)) \
                        or board[r][c] not in [color, EMPTY] \
                        or empty_count == max_empty_count:
                    break

                if board[r][c] == color:
                    _cache_counts[sign*empty_count] += 1
                else:
                    empty_count += 1
                    if empty_count == max_empty_count:
                        break
                    _cache_positions[sign*empty_count] = (r, c)

        return _cache_counts, _cache_positions

    def get_positions(_type, is_player, _cache_counts, _cache_positions,
                      max_empty_count, container=None):
        key = str(_type) + str(_cache_counts[0])
        for i in range(1, max_empty_count):
            for s in [-1, 1]:
                key += str(_cache_counts[s*i])
                key += {None: '0'}.get(_cache_positions[s*i], '1')
        if is_player:
            type_sign = 1
        else:
            type_sign = -1
        key = type_sign * int(key)

        indice = HASHING_TABLE_OF_INDICES.setdefault(key, {None})
        if indice == {None}:
            indice.pop()
            return False, indice

        if container is not None:
            for idx in indice:
                container.add(_cache_positions[idx])
        return True, indice

    def check_five(_position, _color, _move_function):
        _cache_counts, _cache_positions = check(
            _position, _color, _move_function, 1
        )
        if _cache_counts[0] >= 5:
            return True
        else:
            return False

    def check_open_four_and_four(_position, _color, _move_function,
                                 is_open, container=None):
        _cache_counts, _cache_positions = check(
            _position, _color, _move_function, 3, 7
        )

        if is_open:
            gomoku_type = OPEN_FOUR
        else:
            gomoku_type = FOUR

        # the same positions of searching w.r.t the player and the opponent
        hashing_flag, indice = get_positions(
            gomoku_type, True, _cache_counts, _cache_positions,
            3, container
        )
        if not hashing_flag:
            _indice = set()
            for idx in range(1, 3):
                for sign in [-1, 1]:
                    index = sign * idx
                    _cache_position = _cache_positions[index]
                    if _cache_position is None:
                        continue
                    r, c = _cache_position
                    board[r][c] = _color
                    if check_five(_cache_position, _color, _move_function):
                        _indice.add(index)
                    board[r][c] = EMPTY
            if is_open and len(_indice) >= 2:
                for index in _indice:
                    indice.add(index)
            elif not is_open and len(_indice) == 1:
                indice.add(_indice.pop())

        if container is not None:
            for index in indice:
                container.add(_cache_positions[index])

        if is_open and len(indice) >= 2:
            return True
        elif not is_open and len(indice):
            return True
        else:
            return False

    def check_open_three_and_three(_position, _color, _move_function,
                                   is_open, container=None):
        _cache_counts, _cache_positions = check(
            _position, _color, _move_function, 3
        )

        if is_open:
            gomoku_type = OPEN_THREE
        else:
            gomoku_type = THREE

        is_player = (_color == player)

        hashing_flag, indice = get_positions(
            gomoku_type, is_player, _cache_counts, _cache_positions,
            3, container
        )
        if not hashing_flag:
            for idx in range(1, 3):
                for sign in [-1, 1]:
                    index = sign * idx
                    _cache_position = _cache_positions[index]
                    if _cache_position is None:
                        continue
                    r, c = _cache_position
                    if is_player:
                        board[r][c] = player
                        if check_open_four_and_four(_position, _color, _move_function, is_open):
                            indice.add(index)
                        board[r][c] = EMPTY
                    else:
                        flag = False
                        the_only_one = True
                        for _index in [_idx*_sign for _idx in range(1, 3) for _sign in [-1, 1]]:
                            if _index == index or _cache_positions[_index] is None:
                                continue
                            _r, _c = _cache_positions[_index]
                            board[_r][_c] = opponent
                            if check_open_four_and_four(_position, _color, _move_function, is_open):
                                the_only_one = False
                                board[r][c] = player
                                if check_open_four_and_four(_position, _color, _move_function, is_open):
                                    board[r][c] = EMPTY
                                    board[_r][_c] = EMPTY
                                    flag = False
                                    break
                                board[r][c] = EMPTY
                                flag = True
                            board[_r][_c] = EMPTY

                        if flag:
                            indice.add(index)
                        elif the_only_one:
                            board[r][c] = opponent
                            if check_open_four_and_four(_position, _color, _move_function, is_open):
                                indice.add(index)
                            board[r][c] = EMPTY

        if container is not None:
            for index in indice:
                container.add(_cache_positions[index])

        if len(indice):
            return True
        else:
            return False

    def check_open_two_and_two(_position, _color, _move_function,
                               is_open, container=None):
        _cache_counts, _cache_positions = check(
            _position, _color, _move_function, 4
        )

        if is_open:
            gomoku_type = OPEN_TWO
        else:
            gomoku_type = TWO

        is_player = (_color == player)

        hashing_flag, indice = get_positions(
            gomoku_type, is_player, _cache_counts, _cache_positions,
            4, container
        )
        if not hashing_flag:
            for idx in range(1, 4):
                for sign in [-1, 1]:
                    index = sign * idx
                    _cache_position = _cache_positions[index]
                    if _cache_position is None:
                        continue
                    r, c = _cache_position
                    if is_player:
                        board[r][c] = player
                        if check_open_three_and_three(_position, _color, _move_function, is_open):
                            indice.add(index)
                        board[r][c] = EMPTY
                    else:
                        flag = False
                        the_only_one = True
                        for _index in [_idx*_sign for _idx in range(1, 4) for _sign in [-1, 1]]:
                            if _index == index or _cache_positions[_index] is None:
                                continue
                            _r, _c = _cache_positions[_index]
                            board[_r][_c] = opponent
                            if check_open_three_and_three(_position, _color, _move_function, is_open):
                                the_only_one = False
                                board[r][c] = player
                                if check_open_three_and_three(_position, _color, _move_function, is_open):
                                    board[r][c] = EMPTY
                                    board[_r][_c] = EMPTY
                                    flag = False
                                    break
                                board[r][c] = EMPTY
                                flag = True
                            board[_r][_c] = EMPTY

                        if flag:
                            indice.add(index)
                        elif the_only_one:
                            board[r][c] = opponent
                            if check_open_four_and_four(_position, _color, _move_function, is_open):
                                indice.add(index)
                            board[r][c] = EMPTY

        if container is not None:
            for index in indice:
                container.add(_cache_positions[index])

        if len(indice):
            return True
        else:
            return False

    for color in [BLACK, WHITE]:
        _gomoku_types = gomoku_types[color]
        positions = collections.defaultdict(set)

        for op_gt, gt, check_func in [
            (OPEN_FOUR, FOUR, check_open_four_and_four),
            (OPEN_THREE, THREE, check_open_three_and_three),
            (OPEN_TWO, TWO, check_open_two_and_two)
        ]:
            open_gt_set = _gomoku_types[op_gt]
            _positions = positions[op_gt]
            rest_positions_for_searching = set()
            for p in list(open_gt_set):
                flag = False
                for m_f in move_list:
                    if check_func(p, color, m_f, True, _positions):
                        flag = True
                if not flag:
                    open_gt_set.remove(p)
                    rest_positions_for_searching.add(p)

            gt_set = _gomoku_types[gt]
            _positions = positions[gt]
            for p in list(rest_positions_for_searching | gt_set):
                flag = False
                for m_f in move_list:
                    if check_func(p, color, m_f, False, _positions):
                        flag = True
                if flag:
                    gt_set.add(p)
                else:
                    gt_set.discard(p)

        if color == player:
            current_positions = positions
        else:
            opponent_positions = positions

    return (current_positions, opponent_positions), gomoku_types

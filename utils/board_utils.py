import gc
import os
import sys
import json
import random
import collections
from ..constant import *
from .. import path as gomokuZero_path
from memory_profiler import profile


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
OPEN_FOUR = 0
FOUR = 1
OPEN_THREE = 2
THREE = 3
OPEN_TWO = 4
TWO = 5

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
        board._board,
        base_gomoku_types,
        board.history[searched_step:],
        board.player
    )
    board._promising_positions = (current_positions, opponent_positions)
    board._gomoku_types = gomoku_types
    board._searched_step = len(board.history)

    return current_positions, opponent_positions

def _get_promising_positions(board, base_gomoku_types, history, player):
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

    def check(position, color):
        _counts = []
        _positions = []
        for m_f in move_list:
            _cache_counts = {-3: 0, -2: 0, -1: 0, 0: 1, 1: 0, 2: 0, 3: 0}
            _cache_positions = [None] * 7
            for sign in [-1, 1]:
                empty_count = 0
                for delta in range(1, 6):
                    r, c = m_f(position, sign*delta)
                    if not check_border((r, c)) \
                            or board[r][c] not in [color, EMPTY] \
                            or empty_count == 4:
                        break

                    if board[r][c] == color:
                        _cache_counts[empty_count*sign] += 1
                    else:
                        empty_count += 1
                        if empty_count == 4:
                            break
                        _cache_positions[empty_count*sign+3] = (r, c)

            _counts.append(_cache_counts)
            _positions.append(_cache_positions)

        return _counts, _positions

    def get_positions(_type, _counts, _positions, container):
        key = str(_type)
        for _cache_counts, _cache_positions in zip(_counts, _positions):
            key += ''.join([str(_cache_counts[i]) for i in range(-2, 3)])
            for p in _cache_positions:
                key += {None: ' '}.get(p, '_')

        indice = HASHING_TABLE_OF_INDICES.setdefault(key, [None])
        if indice == [None]:
            indice.pop()
            return False, indice
        for idxs, _cache_positions in zip(indice, _positions):
            for idx in idxs:
                container.add(_cache_positions[idx])
        container.discard(None)
        return True, None

    for color_for_searching in [BLACK, WHITE]:

        _gomoku_types = gomoku_types[color_for_searching]
        positions = collections.defaultdict(set)

        is_player = (color_for_searching == player)
        color_sign = 1 if is_player else -1


        #search for open four and four
        open_four_set = _gomoku_types[OPEN_FOUR]
        rest_positions_for_searching = dict()
        for p in list(open_four_set):
            _counts, _positions = check(p, color_for_searching)
            hashing_flag, indice = get_positions(color_sign*OPEN_FOUR, _counts,
                                                 _positions, positions[OPEN_FOUR])
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if cache_counts[0] >= 4 and cache_positions[2] and cache_positions[4]:
                        positions[OPEN_FOUR].add(cache_positions[2])
                        idxs.add(2)
                        positions[OPEN_FOUR].add(cache_positions[4])
                        idxs.add(4)
                    indice.append(idxs)

            if len(positions[OPEN_FOUR]) == 0:
                rest_positions_for_searching[p] = [_counts, _positions]
                open_four_set.remove(p)

        four_set = _gomoku_types[FOUR]
        for p in four_set:
            if p in rest_positions_for_searching:
                continue
            _counts, _positions = check(p, color_for_searching)
            rest_positions_for_searching[p] = [_counts, _positions]
        for p, (_counts, _positions) in rest_positions_for_searching.items():
            tmp_four_positions = set()
            hashing_flag, indice = get_positions(color_sign*FOUR, _counts,
                                                 _positions, tmp_four_positions)
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if cache_counts[0] + cache_counts[-1] >= 4:
                        tmp_four_positions.add(cache_positions[2])
                        idxs.add(2)
                    if cache_counts[0] + cache_counts[1] >= 4:
                        tmp_four_positions.add(cache_positions[4])
                        idxs.add(4)
                    indice.append(idxs)

            tmp_four_positions.discard(None)
            if len(tmp_four_positions):
                positions[FOUR] |= tmp_four_positions
                four_set.add(p)
            else:
                four_set.discard(p)


        #search for open three and three
        open_three_set = _gomoku_types[OPEN_THREE]
        rest_positions_for_searching = dict()
        for p in list(open_three_set):
            _counts, _positions = check(p, color_for_searching)
            tmp_open_three_positions = set()
            hashing_flag, indice = get_positions(color_sign*OPEN_THREE, _counts,
                                                 _positions, tmp_open_three_positions)
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if sum([cache_counts[i] for i in [-1, 0, 1]]) < 3 \
                            or cache_positions[2] is None \
                            or cache_positions[4] is None:
                        indice.append(idxs)
                        continue

                    for sign in [-1, 1]:
                        if cache_counts[0] + cache_counts[sign] >= 3:
                            if cache_positions[2*sign+3] is not None:
                                tmp_open_three_positions.add(cache_positions[sign+3])
                                idxs.add(sign+3)
                                if not is_player \
                                        and cache_counts[0] < 3:
                                    tmp_open_three_positions.add(cache_positions[2*sign+3])
                                    idxs.add(2*sign+3)
                                    tmp_open_three_positions.add(cache_positions[-sign+3])
                                    idxs.add(-sign+3)
                    indice.append(idxs)

            tmp_open_three_positions.discard(None)
            if len(tmp_open_three_positions):
                positions[OPEN_THREE] |= tmp_open_three_positions
            else:
                rest_positions_for_searching[p] = [_counts, _positions]
                open_three_set.remove(p)


        three_set = _gomoku_types[THREE]
        for p in three_set:
            if p in rest_positions_for_searching:
                continue
            _counts, _positions = check(p, color_for_searching)
            rest_positions_for_searching[p] = [_counts, _positions]
        for p, (_counts, _positions) in rest_positions_for_searching.items():
            tmp_three_positions = set()
            hashing_flag, indice = get_positions(color_sign*THREE, _counts,
                                                 _positions, tmp_three_positions)

            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if sum(cache_counts.values()) < 3:
                        indice.append(idxs)
                        continue
                    if cache_counts[0] == 3:
                        if cache_positions[2] and cache_positions[4] \
                                and cache_positions[1] is None \
                                and cache_positions[5] is None:
                            tmp_three_positions.add(cache_positions[2])
                            idxs.add(2)
                            tmp_three_positions.add(cache_positions[4])
                            idxs.add(4)
                        elif cache_positions[2] is None and cache_positions[5]:
                            tmp_three_positions.add(cache_positions[4])
                            idxs.add(4)
                            tmp_three_positions.add(cache_positions[5])
                            idxs.add(5)
                        elif cache_positions[4] is None and cache_positions[1]:
                            tmp_three_positions.add(cache_positions[2])
                            idxs.add(2)
                            tmp_three_positions.add(cache_positions[1])
                            idxs.add(1)

                    elif cache_counts[0] < 3:
                        for sign in [-1, 1]:
                            if cache_counts[0] + cache_counts[sign] == 3:
                                if cache_positions[-sign+3] is None and cache_positions[2*sign+3]:
                                    tmp_three_positions.add(cache_positions[sign+3])
                                    idxs.add(sign+3)
                                    tmp_three_positions.add(cache_positions[2*sign+3])
                                    idxs.add(2*sign+3)
                                elif cache_positions[2*sign+3] is None and cache_positions[sign+3] and cache_positions[-sign+3]:
                                    tmp_three_positions.add(cache_positions[sign+3])
                                    idxs.add(sign+3)
                                    tmp_three_positions.add(cache_positions[-sign+3])
                                    idxs.add(-sign+3)

                            elif cache_counts[sign] == 0 and cache_counts[0] + cache_counts[2*sign] == 3:
                                tmp_three_positions.add(cache_positions[sign+3])
                                idxs.add(sign+3)
                                tmp_three_positions.add(cache_positions[2*sign+3])
                                idxs.add(2*sign+3)

                            elif cache_counts[0] == 1 and cache_counts[sign] == 1 and cache_counts[2*sign] == 1:
                                tmp_three_positions.add(cache_positions[sign+3])
                                idxs.add(sign+3)
                                tmp_three_positions.add(cache_positions[2*sign+3])
                                idxs.add(2*sign+3)

                        if cache_counts[0] == 1 and cache_counts[sign] == 1 and cache_counts[-sign] == 1:
                            tmp_three_positions.add(cache_positions[sign+3])
                            idxs.add(sign+3)
                            tmp_three_positions.add(cache_positions[-sign+3])
                            idxs.add(-sign+3)
                        # if ((cache_counts[0] + cache_counts[-1] >= 3 \
                        #         or cache_counts[0] + cache_counts[-2] >= 3) \
                        #         and cache_positions[1]) \
                        #         or cache_counts[-2] + cache_counts[-1] + cache_counts[0] >= 3:
                        #     tmp_three_positions |= {cache_positions[2], cache_positions[1]}
                        #     idxs.add(2)
                        #     idxs.add(1)
                        # if ((cache_counts[0] + cache_counts[1] >= 3 \
                        #         or cache_counts[0] + cache_counts[2] >= 3) \
                        #         and cache_positions[5]) \
                        #         or cache_counts[0] + cache_counts[1] + cache_counts[2] >= 3:
                        #     tmp_three_positions |= {cache_positions[4], cache_positions[5]}
                        #     idxs.add(4)
                        #     idxs.add(5)
                        #
                        # if cache_counts[-1] + cache_counts[0] + cache_counts[1] >= 3:
                        #     tmp_three_positions |= {cache_positions[2], cache_positions[4]}
                        #     idxs.add(2)
                        #     idxs.add(4)

                    indice.append(idxs)

            tmp_three_positions.discard(None)
            if len(tmp_three_positions):
                positions[THREE] |= tmp_three_positions
                three_set.add(p)
            else:
                three_set.discard(p)


        #search for open two and two
        open_two_set = _gomoku_types[OPEN_TWO]
        rest_positions_for_searching = dict()
        for p in list(open_two_set):
            _counts, _positions = check(p, color_for_searching)
            tmp_open_two_positions = set()
            hashing_flag, indice = get_positions(color_sign*OPEN_TWO, _counts,
                                                 _positions, tmp_open_two_positions)
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if sum(cache_counts.values()) < 2 \
                            or cache_positions[2] is None \
                            or cache_positions[4] is None:
                        indice.append(idxs)
                        continue

                    for sign in [-1, 1]:
                        if cache_counts[0] == 2 and cache_counts[sign] == 0 and cache_counts[2*sign] == 0:
                            if cache_positions[3*sign+3] is not None:
                                tmp_open_two_positions.add(cache_positions[sign+3])
                                idxs.add(sign+3)

                                if is_player:
                                    tmp_open_two_positions.add(cache_positions[2*sign+3])
                                    idxs.add(2*sign+3)
                                else:
                                    if cache_positions[-3*sign+3] is None:
                                        tmp_open_two_positions.add(cache_positions[2*sign+3])
                                        idxs.add(2*sign+3)
                                    if cache_positions[-2*sign+3] is None:
                                        tmp_open_two_positions.add(cache_positions[3*sign+3])
                                        idxs.add(3*sign+3)

                            elif cache_positions[2*sign+3] is not None:
                                if cache_positions[-2*sign+3] is not None:
                                    tmp_open_two_positions.add(cache_positions[sign+3])
                                    idxs.add(sign+3)

                                    if not is_player:
                                        tmp_open_two_positions.add(cache_positions[2*sign+3])
                                        idxs.add(2*sign+3)

                        elif cache_counts[0] == 1 and cache_counts[sign] == 1 and cache_counts[2*sign] == 0:
                            if cache_positions[3*sign+3] is not None:
                                tmp_open_two_positions.add(cache_positions[sign+3])
                                idxs.add(sign+3)
                                tmp_open_two_positions.add(cache_positions[2*sign+3])
                                idxs.add(2*sign+3)
                                if not is_player:
                                    if cache_positions[-2*sign+3] is None:
                                        tmp_open_two_positions.add(cache_positions[3*sign+3])
                                        idxs.add(3*sign+3)

                        elif cache_counts[0] == 1 and cache_counts[sign] == 0 and cache_counts[2*sign] == 1:
                            if cache_positions[3*sign+3] is not None:
                                tmp_open_two_positions.add(cache_positions[sign+3])
                                idxs.add(sign+3)
                                tmp_open_two_positions.add(cache_positions[2*sign+3])
                                idxs.add(2*sign+3)
                                if not is_player:
                                    tmp_open_two_positions.add(cache_positions[3*sign+3])
                                    idxs.add(3*sign+3)

                    indice.append(idxs)

            tmp_open_two_positions.discard(None)
            if len(tmp_open_two_positions):
                positions[OPEN_TWO] |= tmp_open_two_positions
            else:
                rest_positions_for_searching[p] = [_counts, _positions]
                open_two_set.remove(p)


        two_set = _gomoku_types[TWO]
        for p in two_set:
            if p in rest_positions_for_searching:
                continue
            _counts, _positions = check(p, color_for_searching)
            rest_positions_for_searching[p] = [_counts, _positions]
        for p, (_counts, _positions) in rest_positions_for_searching.items():
            tmp_two_positions = set()
            hashing_flag, indice = get_positions(color_sign*TWO, _counts,
                                                 _positions, tmp_two_positions)

            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if sum(cache_counts.values()) < 2:
                        indice.append(idxs)
                        continue

                    for sign in [-1, 1]:
                        if (cache_counts[0] + cache_counts[sign] + cache_counts[2*sign] == 2 \
                                and cache_positions[-sign+3] is None \
                                and cache_positions[3*sign+3] is not None) \
                            or (cache_counts[0] == 1 \
                                    and cache_counts[sign] == 0 \
                                    and cache_counts[2*sign] == 0 \
                                    and cache_counts[3*sign] == 1):
                            tmp_two_positions.add(cache_positions[sign+3])
                            idxs.add(sign+3)
                            tmp_two_positions.add(cache_positions[2*sign+3])
                            idxs.add(2*sign+3)
                            tmp_two_positions.add(cache_positions[3*sign+3])
                            idxs.add(3*sign+3)

                    indice.append(idxs)

            tmp_two_positions.discard(None)
            if len(tmp_two_positions):
                positions[TWO] |= tmp_two_positions
                two_set.add(p)
            else:
                two_set.discard(p)


        if color_for_searching == player:
            current_positions = positions
        else:
            opponent_positions = positions

    return (current_positions, opponent_positions), gomoku_types

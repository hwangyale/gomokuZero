import os
import json
import random
import Queue
import collections
from ..constant import *


try:
    range = xrange
except NameError:
    pass


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

def get_urgent_position(board, move_funcs=[m_f for m_f in move_list]):
    if len(board.history) < 2*(NUMBER-1):
        return None

    _board = board._board
    def get_position_to_win(position, color):
        random.shuffle(move_funcs)
        signs = [-1, 1]
        for m_f in move_funcs:
            counts = {-1: 0, 0: 1, 1: 0}
            four = [None, None]
            random.shuffle(signs)
            for sign in signs:
                empty_flag = False
                for delta in range(1, NUMBER):
                    r, c = m_f(position, sign*delta)
                    if check_border((r, c)):
                        if _board[r][c] == color:
                            if empty_flag:
                                counts[sign] += 1
                            else:
                                counts[0] += 1
                        elif _board[r][c] == EMPTY:
                            if empty_flag:
                                break
                            else:
                                four[(sign+1)//2] = (r, c)
                                empty_flag = True
                        else:
                            break
                    else:
                        break

            for sign in signs:
                if counts[0] + counts[sign] >= (NUMBER-1) \
                        and four[(sign+1)//2] is not None:
                    return four[(sign+1)//2]

        return None

    player = board.player
    position = get_position_to_win(board.history[-2], player)
    if position is not None:
        return position

    opponent = {BLACK: WHITE, WHITE: BLACK}[player]
    position = get_position_to_win(board.history[-1], opponent)
    if position is not None:
        return position

    return None


#hashing
hashing_keys_file = 'hashing_keys_of_size_{:d}.json'.format(SIZE)
if os.path.exists(hashing_keys_file):
    with open(hashing_keys_file, 'r') as f:
        HASHING_KEYS = json.load(f)
else:
    HASHING_KEYS = [
        [
            {color: random.randint(0, 3**(SIZE**2)) for color in [BLACK, WHITE]}
            for _ in range(SIZE)
        ]
        for _ in range(SIZE)
    ]
    with open(hashing_keys_file, 'w') as f:
        json.dump(HASHING_KEYS, f)


#get gomoku types
OPEN_FOUR = 0
FOUR = 1
OPEN_THREE = 2
THREE = 3
OPEN_TWO = 4
TWO = 5

HASHING_TO_POSITIONS_FOR_SEARCHING = {
    0: {
        color: {i: set() for i in range(6)}
        for color in [BLACK, WHITE]
    }
}
HASHING_TABLE_OF_INDICES = dict()

color_mapping = {BLACK: WHITE, WHITE: BLACK}
def get_gomoku_types(hashing_key, board, history):
    gomoku_types = HASHING_TO_POSITIONS_FOR_SEARCHING.setdefault(hashing_key, dict())
    if len(gomoku_types):
        return gomoku_types

    player = [WHITE, PLAYER][len(history) % 2]
    color = color_mapping[player]
    for idx, (r, c) in enumerate(history[::-1], 1):
        hashing_key ^= HASHING_KEYS[r][c][color]
        base_gomoku_types = HASHING_TO_POSITIONS_FOR_SEARCHING.get(hashing_key, None)
        if base_gomoku_types is not None:
            break
        color = color_mapping[color]

    def check(position, color):
        _counts = []
        _positions = []
        for m_f in move_list:
            _cache_counts = {-2: 0, -1: 0, 0: 1, 1: 0, 2: 0}
            _cache_positions = [None] * 5
            for sign in [-1, 1]:
                empty_count = 0
                for delta in range(1, 6):
                    r, c = m_f(position, sign*delta)
                    if not check_border((r, c)) \
                            or board[r][c] not in [color, EMPTY] \
                            or empty_count == 3:
                        break

                    if board[r][c] == color:
                        _cache_counts[empty_count*sign] += 1
                    else:
                        empty_count += 1
                        if empty_count == 3:
                            break
                        else:
                            _cache_positions[empty_count*sign+2] = (r, c)

            _counts.append(_cache_counts)
            _positions.append(_cache_positions)

        return _counts, _positions

    def get_positions(_type, _counts, _positions, container):
        key = str(_type)
        for _cache_counts, _cache_positions in zip(_counts, _positions):
            key += ''.join([str(_cache_counts[i]) for i in range(-2, 3)])
            for p in _cache_positions:
                key += {None: ' '}.get(p, '_')
        indice = HASHING_TABLE_OF_INDICES.get(key, {None})
        if indice == {None}:
            indice.pop()
            return False, indice
        for idxs, _cache_positions in zip(indice, _positions):
            for idx in idxs:
                container.add(_cache_positions[idx])
        container.discard(None)
        return True, None


    for color_for_searching in [BLACK, WHITE]:
        additional_positions = set(history[-idx+(color_for_searching != color)::2])
        _gomoku_types = {
            t: s | additional_positions
            for t, s in base_gomoku_types[color_for_searching].items()
        }

        positions = collections.defaultdict(set)

        open_four_set = _gomoku_types[OPEN_FOUR]
        rest_positions_for_searching = dict()
        for p in list(open_four_set):
            _counts, _positions = check(p, color_for_searching)
            hashing_flag, indice = get_positions(OPEN_FOUR, _counts,
                                                 _positions, positions[OPEN_FOUR])
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if cache_counts[0] >= 4 and cache_positions[1] and cache_positions[3]:
                        positions[OPEN_FOUR].add(cache_positions[1])
                        idxs.add(1)
                        positions[OPEN_FOUR].add(cache_positions[3])
                        idxs.add(3)
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
            hashing_flag, indice = get_positions(FOUR, _counts,
                                                 _positions, tmp_four_positions)
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if cache_counts[0] + cache_counts[-1] >= 4:
                        tmp_four_positions.add(cache_positions[1])
                        idxs.add(1)
                    if cache_counts[0] + cache_counts[1] >= 4:
                        tmp_four_positions.add(cache_positions[3])
                        idxs.add(3)
                    indice.append(idxs)

            if len(tmp_four_positions):
                positions[FOUR] |= tmp_four_positions
                four_set.add(p)
            else:
                four_set.discard(p)


        open_three_set = _gomoku_types[OPEN_THREE]
        rest_positions_for_searching = dict()
        for p in list(open_three_set):
            _counts, _positions = check(p, color_for_searching)
            tmp_open_three_positions = set()
            hashing_flag, indice = get_positions(OPEN_THREE, _counts,
                                                 _positions, tmp_open_three_positions)
            if not hashing_flag:
                for cache_counts, cache_positions in zip(_counts, _positions):
                    idxs = set()
                    if sum([cache_counts[i] for i in [-1, 0, 1]]) < 3 \
                            or cache_positions[1] is None \
                            or cache_positions[3] is None:
                        indice.append(idxs)
                        continue

                    for sign in [-1, 1]:
                        if cache_counts[0] + cache_counts[sign] >= 3:
                            if cache_positions[2*sign+2] is not None:
                                tmp_open_three_positions.add(cache_positions[sign+2])
                                idxs.add(sign+2)
                                if color_for_searching != player \
                                        and cache_counts[0] < 3:
                                    tmp_open_three_positions.add(cache_positions[2*sign+2])
                                    idxs.add(2*sign+2)
                                    tmp_open_three_positions.add(cache_positions[-sign+2])
                                    idxs.add(-sign+2)
                    indice.append(idxs)

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
            hashing_flag, indice = get_positions(THREE, _counts,
                                                 _positions, tmp_three_positions)
            for cache_counts, cache_positions in zip(_counts, _positions):
                idxs = set()
                if sum(cache_counts.values()) < 3:
                    indice.append(idxs)
                    continue
                if cache_counts[0] == 3:
                    if cache_positions[1] and cache_positions[3]:
                        tmp_three_positions |= {cache_positions[1], cache_positions[3]}
                        idxs.add(1)
                        idxs.add(3)

                else:
                    if ((cache_counts[0] + cache_counts[-1] >= 3 \
                            or cache_counts[0] + cache_counts[-2] >= 3) \
                            and cache_positions[0]) \
                            or cache_counts[-2] + cache_counts[-1] + cache_counts[0] >= 3:
                        tmp_three_positions |= {cache_positions[1], cache_positions[0]}
                        idxs.add(1)
                        idxs.add(0)
                    if ((cache_counts[0] + cache_counts[1] >= 3 \
                            or cache_counts[0] + cache_counts[2] >= 3) \
                            and cache_positions[4]) \
                            or cache_counts[0] + cache_counts[1] + cache_counts[2] >= 3:
                        tmp_three_positions |= {cache_positions[3], cache_positions[4]}
                        idxs.add(3)
                        idxs.add(4)

                    if cache_counts[-1] + cache_counts[0] + cache_counts[1] >= 3:
                        tmp_three_positions |= {cache_positions[1], cache_positions[3]}
                        idxs.add(1)
                        idxs.add(3)

                indice.append(idxs)

            tmp_three_positions.discard(None)
            if len(tmp_three_positions):
                positions[THREE] |= tmp_three_positions
                three_set.add(p)
            else:
                three_set.discard(p)


        open_two_set = _gomoku_types[OPEN_TWO]
        two_set = _gomoku_types[TWO]

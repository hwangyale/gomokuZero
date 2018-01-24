import collections
from ..constant import *
from .board_utils import check_border, move_list

BOARDS_TO_THREATS = collections.defaultdict(collections.defaultdict(set))

def get_threats(board):
    if len(board.history) < 6:
        return None

    _board = board._board

    def check(position, color, to_block):
        positions = set()
        for m_f in move_list:
            counts = {-1: 0, 0: 1, 1: 0}
            cache_positions = [None]*4
            for sign in [-1, 1]:
                empty_flag = 0
                for delta in range(1, NUMBER):
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

            if sum(counts.values()) < 3 or positions[0] is None or positions[2] is None:
                continue

            for sign in [-1, 1]:
                if counts[0] + counts[sign] >= 3:
                    if cache_positions[sign+2] is not None:
                        positions.add(cache_positions[sign+1])
                        if to_block and counts[0] < 3:
                            positions.add(cache_positions[sign+2])

        return positions

    all_positions = set()

    player = board.player
    opponent = {BLACK: WHITE, WHITE: BLACK}[player]

    for color, to_block_flag, idx in [(player, False, -1), (opponent, True, -2)]:
        position_set = BOARDS_TO_THREATS[board][color]
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

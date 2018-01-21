import random
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

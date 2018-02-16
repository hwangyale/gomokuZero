from __future__ import print_function

import sys
from gomokuZero.utils.faster_board_utils import *
from gomokuZero.board.board import Board

try:
    input = raw_input
except NameError:
    pass

# history = [(8, 7), (9, 6), (11, 6), (11, 7), (7, 7), (6, 7),
#            (9, 8), (8, 8), (7, 6), (10, 9), (8, 3), (7, 4)]
#
# board = Board()
# for r, c in history:
#     print(board)
#     print(get_promising_positions(board)[0])
#     print(get_promising_positions(board)[1])
#     print('\n')
#     board.move((r, c))

def run(check_type, check_promising_positions, check_gomoku_types):
    board = Board()
    while not board.is_over:
        current_positions, opponent_positions = get_promising_positions(board)

        history = board.history
        copy_board = Board(history)

        if check_promising_positions:
            copy_current_positions, copy_opponent_positions = get_promising_positions(copy_board)

            for original_positions, copy_positions in [
                (current_positions, copy_current_positions),
                (opponent_positions, copy_opponent_positions)
            ]:
            # for original_positions, copy_positions in [
            #     (current_positions, copy_current_positions)
            # ]:
                for gt in GOMOKU_TYPES:
                    if gt == check_type and original_positions[gt] != copy_positions[gt]:
                        print(board)
                        print('gomoku type: {}\n'.format(
                            {
                                OPEN_FOUR: 'open four',
                                FOUR: 'four',
                                OPEN_THREE: 'open three',
                                THREE: 'three',
                                OPEN_TWO: 'open two',
                                TWO: 'two'
                            }[gt]
                        ))
                        print({
                            id(current_positions): 'current positions\n',
                            id(opponent_positions): 'opponent positions\n'
                        }[id(original_positions)])
                        print('{}\n\n is different from \n\n{}\n'.format(
                            original_positions[gt], copy_positions[gt]
                        ))
                        tmp = original_positions[gt] - copy_positions[gt]
                        if len(tmp):
                            print('{} of original positions\n'.format(tmp))
                        tmp = copy_positions[gt] - original_positions[gt]
                        if len(tmp):
                            print('{} of copy positions\n'.format(tmp))

                        player = board.player
                        opponent = {BLACK: WHITE, WHITE: BLACK}[player]

                        if original_positions == current_positions:
                            _gomoku_types = board._gomoku_types[player]
                        else:
                            _gomoku_types = board._gomoku_types[opponent]
                        print('the gomoku type positions: {}\n'.format(_gomoku_types[gt]))

                        if copy_positions == copy_current_positions:
                            _copy_gomoku_types = copy_board._gomoku_types[player]
                        else:
                            _copy_gomoku_types = copy_board._gomoku_types[opponent]
                        print('the copy gomoku type positions: {}\n'.format(_copy_gomoku_types[gt]))

                        print('different gomoku type positions:{}\n'.format(
                            _copy_gomoku_types[gt] - _gomoku_types[gt]
                        ))
                        exit()

        if check_gomoku_types:
            gomoku_types = board._gomoku_types
            copy_gomoku_types = copy_board._gomoku_types
            for color in [BLACK, WHITE]:
                _gomoku_types = gomoku_types[color]
                _copy_gomoku_types = copy_gomoku_types[color]
                for gt in GOMOKU_TYPES:
                    if gt != check_type:
                        continue
                    if _gomoku_types[gt] != _copy_gomoku_types[gt]:
                        print(board)
                        print('history: {}\n'.format(board.history))
                        print('color: {}\n'.format({BLACK: 'black', WHITE: 'white'}[color]))
                        print('gomoku type: {}\n'.format(
                            {
                                OPEN_FOUR: 'open four',
                                FOUR: 'four',
                                OPEN_THREE: 'open three',
                                THREE: 'three',
                                OPEN_TWO: 'open two',
                                TWO: 'two'
                            }[gt]
                        ))
                        print('{}\n\n is different from \n\n{}\n'.format(
                            _gomoku_types[gt], _copy_gomoku_types[gt]
                        ))
                        tmp = _gomoku_types[gt] - _copy_gomoku_types[gt]
                        if len(tmp):
                            print('{} of original gomoku types\n'.format(tmp))
                        tmp = _copy_gomoku_types[gt] - _gomoku_types[gt]
                        if len(tmp):
                            print('{} of copy gomoku types\n'.format(tmp))
                        exit()


        position = random.choice(list(board.legal_positions))
        board.move(position)

if __name__ == '__main__':
    for _ in range(10):
        run(OPEN_THREE, True, False)

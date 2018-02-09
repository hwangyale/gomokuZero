from ..constant import *
from ..utils import tolist
from ..utils.board_utils import *


try:
    range = xrange
except NameError:
    pass


class Board(object):
    def __init__(self, history=[], original_board=None):
        self.reset()
        for position in history[:-1]:
            self.move(position, check_flag=False)
        for position in history[-1:]:
            self.move(position, check_flag=True)
        self.original_board = original_board

    @property
    def player(self):
        return self._player

    @property
    def is_over(self):
        return self.winner is not None

    def move(self, position, check_flag=True):
        if check_flag:
            if position not in self.legal_positions:
                raise Exception('illegal position:'+str(position))
        self._board[position[0]][position[1]] = self.player
        try:
            self.legal_positions.remove(position)
        except:
            print self.history
        self._player = {BLACK: WHITE, WHITE: BLACK}[self._player]
        self.last_position = position
        self.history.append(position)
        if check_flag:
            return self.get_winner()
        return None

    def get_winner(self):
        board = self._board
        position = self.last_position
        player = board[position[0]][position[1]]
        for m_f in move_list:
            count = 1
            five = [position, position]
            for sign in [-1, 1]:
                for delta in range(1, NUMBER):
                    r, c = m_f(position, sign*delta)
                    if check_border((r, c)) and board[r][c] == player:
                        count += 1
                        five[(sign+1)//2] = (r, c)
                    else:
                        break
            if count >= NUMBER:
                self.winner = player
                self.five = five
                return player
        self.winner = None if self.legal_positions else DRAW
        self.five = []
        return self.winner

    def __getitem__(self, k):
        try:
            return self._board[k[0]][k[1]]
        except:
            raise Exception('Unknow key:{:s}'.format(str(k)))

    def __str__(self):
        s = ' '*4 + ' '.join(['{:<2d}'.format(j+1) for j in range(SIZE)]) + '\n'
        for i, row in enumerate(self._board, 1):
            s += '{:<2d}  '.format(i)
            for v in row:
                s += {EMPTY: '_', BLACK: 'X', WHITE: 'O'}[v] + '  '
            s += '\n'
        return s

    def copy(self):
        return self.__class__(self.history, self)

    def reset(self):
        self.history = []
        self._board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
        self.legal_positions = set((r, c) for r in range(SIZE) for c in range(SIZE))
        self._player = BLACK
        self.last_position = None
        self.winner = None
        self.five = []

    def get_config(self):
        return {'history': self.history}

    @classmethod
    def instantiate_from_config(cls, config):
        return cls(**config)

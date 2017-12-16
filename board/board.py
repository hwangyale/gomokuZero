from ..common import *


try:
    range = xrange
except NameError:
    pass


class Board(object):
    def __init__(self, history=[]):
        self.reset()
        for action in history[:-1]:
            self.move(action, check_flag=False)
        for action in history[-1:]:
            self.move(action, check_flag=True)

    @property
    def player(self):
        return self._player

    @property
    def is_over(self):
        return self.winner is not None

    def move(self, action, check_flag=True):
        if check_flag:
            if action not in self.legal_actions:
                raise Exception('illegal action:'+str(action))
        self._board[action[0]][action[1]] = self.player
        self.legal_actions.remove(action)
        self._player = {BLACK: WHITE, WHITE: BLACK}[self._player]
        self.last_action = action
        self.history.append(action)
        if check_flag:
            return self.get_winner()
        return None

    def get_winner(self):
        board = self._board
        action = self.last_action
        player = board[action[0]][action[1]]
        for m_f in move_list:
            count = 1
            winner_pos = [action, action]
            for sign in [-1, 1]:
                for delta in range(1, NUMBER):
                    r, c = m_f(action, sign*delta)
                    if check_border((r, c)) and board[r][c] == player:
                        count += 1
                        winner_pos[(sign+1)/2] = (r, c)
                    else:
                        break
            if count >= NUMBER:
                self.winner = player
                self.winner_pos = winner_pos
                return player
        self.winner = None if self.legal_actions else DRAW
        self.winner_pos = []
        return self.winner

    def __getitem__(self, k):
        if not isinstance(k, tuple) or len(k) != 2:
            raise Exception('Unknow key:{:s}'.format(str(k)))
        return self._board[k[0]][k[1]]

    def __str__(self):
        s = ''
        for row in self._board:
            for v in row:
                s += str(v) + ' '
            s += '\n'
        return s

    __repr__ = __str__

    def copy(self):
        return self.__class__(self.history)

    def reset(self):
        self.history = []
        self._board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
        self.legal_actions = set((r, c) for r in range(SIZE) for c in range(SIZE))
        self._player = BLACK
        self.last_action = None
        self.winner = None
        self.winner_pos = []

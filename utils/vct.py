import time
import Queue
import random
import threading
import collections
from .board_utils import get_promising_positions
from .board_utils import OPEN_FOUR, FOUR, OPEN_THREE, THREE, OPEN_TWO, TWO
from . import tolist

OR = 0
AND = 1

INF = 10**7

class Node(object):
    def __init__(self, node_type, board, depth, parent=None, value=None):
        self.node_type = node_type
        self.board = board
        self.depth = depth
        self.parent = parent
        self.value = value
        self.children = dict()
        self.expanded = False
        self.selected_node = None
        self.set_proof_and_disproof()

    def set_proof_and_disproof(self):
        if self.expanded:
            selected_node = None
            if self.node_type:
                proof = 0
                disproof = INF
                items = list(self.children.items())
                for position, node in items:
                    proof += node.proof
                    if disproof > node.disproof:
                        disproof = node.disproof
                        selected_node = (position, node)
                    # if node.proof == 0 or node.disproof == 0:
                        # del self.children[position]
            else:
                proof = INF
                disproof = 0
                items = list(self.children.items())
                for position, node in items:
                    disproof += node.disproof
                    if proof > node.proof:
                        proof = node.proof
                        selected_node = (position, node)
                    # if node.proof == 0 or node.disproof == 0:
                        # del self.children[position]
            self.proof = proof
            self.disproof = disproof
            self.selected_node = selected_node

        else:
            if self.value is None:
                self.proof = 1
                self.disproof = 1
            elif self.value:
                self.proof = 0
                self.disproof = INF
            else:
                self.proof = INF
                self.disproof = 0
        return self.proof, self.disproof

    def develop(self, positions2board_values):
        assert not self.expanded, '{:d} {:d} {}'.format(self.proof, self.disproof, self.children)
        node_type = self.node_type ^ 1
        depth = self.depth + 1
        print depth, positions2board_values.keys()
        for position, (board, value) in positions2board_values.items():
            self.children[position] = Node(node_type, board, depth, self, value)
        self.expanded = True

    def update(self):
        old_proof = self.proof
        old_disproof = self.disproof
        proof, disproof = self.set_proof_and_disproof()
        if self.parent is not None and (proof != old_proof or disproof != old_disproof):
            return self.parent.update()
        return self


class VCT(threading.Thread):
    def __init__(self, board, container, lock, max_depth, max_time):
        self.board = board
        self.container = container
        self.lock = lock
        self.max_depth = max_depth
        self.max_time = max_time
        super(VCT, self).__init__()

    def run(self):
        board = self.board
        container = self.container
        lock = self.lock
        max_depth = self.max_depth
        max_time = self.max_time

        if len(board.history) < 6:
            lock.acquire()
            container[board] = (False, [])
            lock.release()
            return

        player = board.player

        def evaluate(_board, _depth):
            unknown = None if _depth < max_depth else False

            lock.acquire()
            _current_positions, _opponent_positions = get_promising_positions(_board)
            lock.release()
            if len(_current_positions[OPEN_FOUR]) or len(_current_positions[FOUR]):
                value = _board.player == player
                _positions = list(_current_positions[OPEN_FOUR] | _current_positions[FOUR])
            elif len(_opponent_positions[OPEN_FOUR]) or len(_opponent_positions[FOUR]) > 1:
                value = _board.player != player
                _positions = list(_opponent_positions[OPEN_FOUR] | _opponent_positions[FOUR])
            elif len(_opponent_positions[FOUR]) == 1:
                if _board.player == player:
                    _positions = (_current_positions[OPEN_TWO] \
                                 | _current_positions[THREE]) \
                                 & _opponent_positions[FOUR]
                    if len(_positions):
                        value = unknown
                        _positions = list(_positions)
                    else:
                        value = False
                        _positions = list(_opponent_positions[FOUR])
                else:
                    value = unknown
                    _positions = list(_opponent_positions[FOUR])
            elif len(_current_positions[OPEN_THREE]):
                value = _board.player != player
                _positions = list(_current_positions[OPEN_THREE])

            elif _board.player == player:
                if len(_opponent_positions[OPEN_THREE]):
                    _positions = (_opponent_positions[OPEN_THREE] \
                                 & _current_positions[OPEN_TWO]) \
                                 | _current_positions[THREE]
                    if len(_positions):
                        value = unknown
                        _positions = list(_positions)
                    else:
                        value = False
                        _positions = list(_opponent_positions[OPEN_THREE] | _current_positions[THREE])
                else:
                    _positions = _current_positions[OPEN_TWO] | _current_positions[THREE]
                    if len(_positions):
                        value = unknown
                        _positions = list(_positions)
                    else:
                        value = False
                        _positions = []
            else:
                if len(_opponent_positions[OPEN_THREE]):
                    value = unknown
                    _positions = list(_opponent_positions[OPEN_THREE] | _current_positions[THREE])
                else:
                    value = False
                    _positions = []

            return value, _positions


        value, positions = evaluate(board, 0)
        if value is None:
            root = Node(AND, board, 0, None, value)
        else:
            lock.acquire()
            if value:
                container[board] = (True, positions)
            else:
                container[board] = (False, [])
            lock.release()
            return

        boards2positions = {board: positions}
        def developNode(_node):
            _board = _node.board
            _positions2board_values = dict()
            _depth = _node.depth + 1
            for _position in boards2positions[_board]:
                child_board = _board.copy()
                child_board.move(_position)
                _value, child_positions = evaluate(child_board, _depth)
                boards2positions[child_board] = child_positions
                _positions2board_values[_position] = (child_board, _value)
            _node.develop(_positions2board_values)


        depth = 0
        start = time.time()
        node = root
        while root.proof != 0 and root.disproof != 0 \
                and (time.time() - start) < max_time:
            while node.selected_node is not None:
                node = node.selected_node[1]
            developNode(node)
            node = node.update()

        lock.acquire()
        if root.proof == 0:
            container[board] = (True, [root.selected_node[0]])
        else:
            container[board] = (False, [])
        lock.release()

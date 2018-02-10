import gc
import sys
import time
import Queue
import random
import collections
from .board_utils import get_promising_positions, get_hashing_key_of_board
from .board_utils import OPEN_FOUR, FOUR, OPEN_THREE, THREE, OPEN_TWO, TWO
from .thread_utils import lock, Thread
from .vct_utils import plot
from . import tolist
from ..constant import *
from memory_profiler import profile

OR = 0
AND = 1

INF = 10**7

HASHING_TABLE_OF_VCT = dict()

class Node(object):
    count = 0
    def __init__(self, node_type, board, depth,
                 cache_hashing_table_of_vct,
                 parent=None, value=None):
        self.node_type = node_type
        self.board = board
        self.depth = depth
        self.parent = parent
        self.value = value
        self.children = dict()
        self.expanded = False
        self.selected_node = None
        self.cache_hashing_table_of_vct = cache_hashing_table_of_vct
        self.set_proof_and_disproof()

        self.__class__.count += 1

    def set_proof_and_disproof(self):
        if self.expanded:
            selected_node = None
            if self.node_type:
                proof = 0
                disproof = INF
                items = list(self.children.items())
                for position, node in items:
                    proof += node.proof
                    if disproof >= node.disproof:
                        disproof = node.disproof
                        selected_node = (position, node)
                    if node.proof == 0 or node.disproof == 0:
                        del self.children[position]
            else:
                proof = INF
                disproof = 0
                items = list(self.children.items())
                for position, node in items:
                    disproof += node.disproof
                    if proof >= node.proof:
                        proof = node.proof
                        selected_node = (position, node)
                    if node.proof == 0 or node.disproof == 0:
                        del self.children[position]
            if proof >= INF:
                disproof = 0
            elif disproof >= INF:
                proof = 0
            self.proof = proof
            self.disproof = disproof
            self.selected_node = selected_node
            if self.node_type == OR:
                if proof == 0 or disproof == 0:
                    key = get_hashing_key_of_board(self.board)
                    if proof == 0:
                        HASHING_TABLE_OF_VCT[key] = selected_node[0]
                    else:
                        self.cache_hashing_table_of_vct[key] = True

        else:
            if self.value is None:
                self.proof = 1
                self.disproof = 1 + self.depth // 2
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
        for position, (board, value) in positions2board_values.items():
            self.children[position] = Node(node_type, board, depth,
                                           self.cache_hashing_table_of_vct,
                                           self, value)
        self.expanded = True

    def update(self):
        old_proof = self.proof
        old_disproof = self.disproof
        proof, disproof = self.set_proof_and_disproof()
        if self.parent is not None and (proof != old_proof or disproof != old_disproof):
            return self.parent.update()
        return self


class VCT(Thread):
    def __init__(self, board, container, lock, max_depth, max_time, locked,
                 global_threat=True, included_four=True):
        self.board = board
        self.container = container
        self.lock = lock
        self.max_depth = max_depth
        self.max_time = max_time
        self.locked = locked
        self.global_threat = global_threat
        self.included_four = included_four
        self.cache_hashing_table_of_vct = dict()
        super(VCT, self).__init__()

    def run(self):
        board = self.board
        container = self.container
        lock = self.lock
        max_depth = self.max_depth
        max_time = self.max_time
        locked = self.locked

        if len(board.history) < 6:
            if locked:
                container[board] = (False, [])
            else:
                lock.acquire()
                container[board] = (False, [])
                lock.release()
            return

        player = board.player

        def evaluate(_board, _depth):
            board_key = get_hashing_key_of_board(_board)
            _position = HASHING_TABLE_OF_VCT.get(board_key, None)
            if _position is not None:
                return True, [_position], []
            elif self.cache_hashing_table_of_vct.get(board_key, False):
                return False, [], []

            unknown = None if _depth < max_depth else False

            if locked:
                _current_positions, _opponent_positions = get_promising_positions(_board)
            else:
                lock.acquire()
                _current_positions, _opponent_positions = get_promising_positions(_board)
                lock.release()

            if len(_current_positions[OPEN_FOUR]) or len(_current_positions[FOUR]):
                value = _board.player == player
                _positions = list(_current_positions[OPEN_FOUR] | _current_positions[FOUR])
                position_values = []

            elif len(_opponent_positions[OPEN_FOUR]) or len(_opponent_positions[FOUR]) > 1:
                value = _board.player != player
                _positions = list(_opponent_positions[OPEN_FOUR] | _opponent_positions[FOUR])
                position_values = []

            elif len(_opponent_positions[FOUR]) == 1:
                if _board.player == player:
                    if len(_current_positions[OPEN_THREE]):
                        value = unknown
                        _positions = list(_opponent_positions[FOUR])
                    else:
                        _positions = (_current_positions[OPEN_TWO] | _current_positions[THREE]) \
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
                position_values = []

            elif len(_current_positions[OPEN_THREE]):
                value = _board.player == player
                _positions = list(_current_positions[OPEN_THREE])
                position_values = []

            elif _board.player == player:
                if len(_opponent_positions[OPEN_THREE]):
                    if self.included_four:
                        _positions = (_opponent_positions[OPEN_THREE] \
                                     & _current_positions[OPEN_TWO]) \
                                     | _current_positions[THREE]
                    else:
                        _positions = _opponent_positions[OPEN_THREE] & _current_positions[OPEN_TWO]
                    if len(_positions):
                        value = unknown
                        _positions = list(_positions)
                    else:
                        value = False
                        if self.included_four:
                            _positions = list(_opponent_positions[OPEN_THREE] | _current_positions[THREE])
                        else:
                            _positions = list(_opponent_positions[OPEN_THREE])
                    position_values = []
                else:
                    _positions = list(_current_positions[OPEN_TWO])
                    position_values = [1] * len(_positions)
                    if self.included_four:
                        for p in _current_positions[THREE]:
                            if p not in _current_positions[OPEN_TWO]:
                                _positions.append(p)
                                position_values.append(FOUR_DELAY)
                    if len(_positions):
                        value = unknown
                    else:
                        value = False
                        _positions = []
                        position_values = []
            else:
                if len(_opponent_positions[OPEN_THREE]):
                    value = unknown
                    if self.global_threat:
                        _positions = list(_opponent_positions[OPEN_THREE] | _current_positions[THREE])
                    else:
                        _positions = list(_opponent_positions[OPEN_THREE])
                else:
                    value = False
                    _positions = []
                position_values = []

            return value, _positions, position_values


        value, positions, position_values = evaluate(board, 0)
        if value is None:
            root = Node(OR, board, 0,
                        self.cache_hashing_table_of_vct,
                        None, value)
        else:
            if locked:
                if value:
                    container[board] = (True, positions)
                else:
                    container[board] = (False, [])
            else:
                lock.acquire()
                if value:
                    container[board] = (True, positions)
                else:
                    container[board] = (False, [])
                lock.release()
            return

        boards2positions = {board: (positions, position_values)}
        def developNode(_node):
            _board = _node.board
            _positions2board_values = dict()
            _depth = _node.depth + 1
            _positions, _position_values = boards2positions[_board]
            for _position in _positions:
                child_board = _board.copy()
                child_board.move(_position, check_flag=False)
                child_value, child_positions, child_position_values = evaluate(child_board, _depth)
                boards2positions[child_board] = (child_positions, child_position_values)
                _positions2board_values[_position] = (child_board, child_value)
            _node.develop(_positions2board_values)
            if len(_position_values):
                for _position, _position_value in zip(_positions, _position_values):
                    _node.children[_position].proof = _position_value

        depth = 0
        start = time.time()
        node = root
        count = 0
        while root.proof != 0 and root.disproof != 0 \
                and (time.time() - start) < max_time:
            while node.selected_node is not None:
                node = node.selected_node[1]
            developNode(node)
            node = node.update()
            count += 1
            # plot(root, 'vct_tree_searching_time_{:d}.png'.format(count))
        # plot(root, 'vct_tree_searching_time.png')

        if locked:
            if root.proof == 0:
                container[board] = (True, [root.selected_node[0]])
            else:
                container[board] = (False, [])
        else:
            lock.acquire()
            if root.proof == 0:
                container[board] = (True, [root.selected_node[0]])
            else:
                container[board] = (False, [])
            lock.release()


def get_vct(boards, max_depth, max_time, max_thread=10, locked=False, global_threat=True,
            included_four=True):
    boards = tolist(boards)
    number = len(boards)
    container = dict()
    vcts = dict()
    count = 0
    search_again = []
    while vcts or count < number:
        while len(vcts) < max_thread and count < number:
            board = boards[count].copy()
            if included_four:
                tmp_included_four = False
                tmp_max_time = max_time / 2.0
            else:
                tmp_included_four = included_four
                tmp_max_time = max_time
            vct = VCT(board, container, lock, max_depth, tmp_max_time, locked=locked,
                      global_threat=global_threat, included_four=tmp_included_four)
            if locked:
                vct.run()
            else:
                vct.start()
            vcts[board] = vct
            count += 1

        time.sleep(0.001)
        cache_boards = list(vcts.keys())
        if locked:
            for board in cache_boards:
                if board in container:
                    if included_four:
                        if not container[board][0]:
                            search_again.append(board.original_board)
                    container[board.original_board] = container.pop(board)
                    del vcts[board]
        else:
            lock.acquire()
            for board in cache_boards:
                if board in container:
                    if included_four:
                        if not container[board][0]:
                            search_again.append(board.original_board)
                    container[board.original_board] = container.pop(board)
                    del vcts[board]
            lock.release()

    if len(search_again):
        number = len(search_again)
        vcts = dict()
        count = 0
        while vcts or count < number:
            while len(vcts) < max_thread and count < number:
                board = search_again[count].copy()
                vct = VCT(board, container, lock, max_depth, max_time/2.0, locked=locked,
                          global_threat=global_threat, included_four=True)
                if locked:
                    vct.run()
                else:
                    vct.start()
                vcts[board] = vct
                count += 1
            time.sleep(0.001)
            cache_boards = list(vcts.keys())
            if locked:
                for board in cache_boards:
                    if board in container:
                        container[board.original_board] = container.pop(board)
                        del vcts[board]
            else:
                lock.acquire()
                for board in cache_boards:
                    if board in container:
                        container[board.original_board] = container.pop(board)
                        del vcts[board]
                lock.release()

    vct_results = [container[board] for board in boards]
    if len(boards) == 1:
        return vct_results[0]
    else:
        return vct_results

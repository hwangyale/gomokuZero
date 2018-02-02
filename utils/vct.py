import time
import Queue
import random
import threading
import collections
from .board_utils import get_promising_positions
from . import tolist

OR = 0
AND = 1

INF = 10**7

class Node(object):
    def __init__(self, node_type, parent=None, value=None):
        self.node_type = node_type
        self.parent = parent
        self.value = value
        self.children = dict()
        self.expanded = False
        self.selected_node = None

    def set_proof_and_disproof(self):
        if self.expanded:
            selected_node = None
            if self.node_type
                proof = 0
                disproof = INF
                for position, node in self.children.items():
                    proof += node.proof
                    if disproof > node.disproof:
                        disproof = node.disproof
                        selected_node = (position, node)
            else:
                proof = INF
                disproof = 0
                for position, node in self.children.items():
                    disproof += node.disproof
                    if proof > node.proof:
                        proof = node.proof
                        selected_node = (position, node)
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

    def develop(self, positions2values):
        assert not self.expanded
        node_type = self.node_type ^ 1
        for position, value in positions2values.items():
            self.children[position].append(Node(self, node_type, value))
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
            container[board] = []
            lock.release()
            return

        player = board.player
        start = time.time()

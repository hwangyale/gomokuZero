import random
import threading
import collections
from .board_utils import get_promising_positions

OR = 0
AND = 1

INF = 10**7

class Node(object):
    def __init__(self, node_type, parent=None, value=None):
        self.node_type = node_type
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

        elif self.evaluated:
            if self.value is None:
                self.proof = 1
                self.disproof = 1
            elif self.value:
                self.proof = INF
                self.disproof = 0
            else:
                self.proof = 0
                self.disproof = INF
        else:
            self.proof = 1
            self.disproof = 1
        return self.proof, self.disproof

    def develop(self, positions2values):
        assert not self.expanded
        node_type = self.node_type ^ 1
        if node_type:
            for position, value in positions2values.items():
                node = Node(self, node_type, value)
                self.children[position].append(node)
        else:
            for position, value in positions2values.items():
                node = Node(self, node_type, value)
                self.children[position].append(node)
        self.expanded = True

    def update(self):
        old_proof = self.proof
        old_disproof = self.disproof
        proof, disproof = self.set_proof_and_disproof()
        if self.parent and (proof != old_proof or disproof != old_disproof):
            return self.parent.update()
        return self

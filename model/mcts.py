from ..constant import *

class Node(object):
    def __init__(self, prior=0.0, parent=None, children=None):
        self.parent = None
        if children is None:
            self.children = {}
        else:
            self.children = children

        self.P = prior
        self.N = 0.0
        self.W = 0.0

    @property
    def Q(self):
        if self.N:
            return self.W / self.N
        return 0.0

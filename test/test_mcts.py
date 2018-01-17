from __future__ import print_function
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS
from gomokuZero.board.board import Board

pvn = PolicyValueNetwork(blocks=3, filters=16)
mcts = MCTS(pvn)
mcts.get_policies(Board(), 1.0, 1000, 1)
root = mcts.boards2roots.values()[0]

count = 0

def visit(node, depth):
    if node.is_virtual:
        global count
        count += 1
        print(count, len(node.is_virtual), depth)
    for child_node in node.children.values():
        visit(child_node, depth+1)

visit(root, 0)

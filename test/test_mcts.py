from __future__ import print_function
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS
from gomokuZero.board.board import Board
import time

pvn = PolicyValueNetwork(blocks=1, filters=32, create_function_name='create_resnet_version_3')
mcts = MCTS(pvn)
start = time.time()
mcts.get_positions(Board(), 1.0, 100, 2, exploration_epsilon=0.25, gamma=0.0,
                   max_depth=4, verbose=2)
end = time.time()
print('time:{:.4f}'.format(end-start))
root = list(mcts.boards2roots.values())[0]

count = 0

def visit(node, depth):
    if node.is_virtual:
        global count
        count += 1
        print(count, len(node.is_virtual), depth)
    for child_node in node.children.values():
        visit(child_node, depth+1)

visit(root, 0)

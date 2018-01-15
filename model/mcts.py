import threading
import warnings

from ..constant import *
from ..utils import tolist

class Node(object):
    def __init__(self, prior=1.0, parent=None, children=None):
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

    def U(self, total_N):
        return C_PUCT * self.P * total_N**0.5 / (1 + self.N)

    def value(self, total_N):
        return self.Q + self.U(total_N)

    def select(self):
        '''lock the thread
        '''
        if self.children:
            children = self.children
            total_N = max(sum([n.N for n in children.values()]), 1.0)
            position, node = max(children.items(), key=lambda (p, n): n.value(total_N))
            node.W -= VIRTUAL_LOSS
            node.N += VIRTUAL_VISIT
            return position, node
        else:
            raise Exception('No children to select')

    def expand(self, policy):
        '''lock the thread
        '''
        if self.children:
            raise Exception('Expand the expanded node')
        for l_p, prob in policy.iteritems():
            self.children[l_p] = self.__class__(prob, self)

    def backup(self, value):
        '''lock the thread
        '''
        self.W += value
        self.N += 1.0
        if self.parent:
            self.parent.backup(-value)


class SearchThread(threading.Thread):
    def __init__(self, root, board, lock, container, name=None):
        self.root = root
        self.board = board
        self.lock = lock
        self.container = container
        super(SearchThread, self).__init__(name=name)

    def run(self):
        node = self.root
        board = self.board
        lock = self.lock

        lock.acquire()
        if not node.children:
            container.append((node, board))
            lock.release()
        else:
            lock.release()
            while not board.is_over():
                lock.acquire()
                position, node = node.select()
                if not node.children:
                    board.move(position)
                    container.append((node, board))
                    lock.release()
                    break
                lock.release()
                board.move(position)


class MCTS(object):
    def __init__(self, rollout_time=100, max_thread=8, evaluation_size=8):
        if max_thread < evaluation_size:
            warnings.warn('`max_thread` is less than `evaluation_size`, '
                          'setting: `evaluation_size = max_thread`')
            evaluation_size = max_thread
        self.rollout_time = rollout_time
        self.max_thread = max_thread
        self.evaluation_size = evaluation_size

        self.boards2roots = {}

    def rollout(self, boards,
                rollout_time=None, max_thread=None,
                evaluation_size=None):
        if rollout_time is None:
            rollout_time = rollout_time
        if max_thread is None:
            max_thread = max_thread
        if evaluation_size is None:
            evaluation_size = evaluation_size
        if max_thread < evaluation_size:
            warnings.warn('`max_thread` is less than `evaluation_size`, '
                          'setting: `evaluation_size = max_thread`')
            evaluation_size = max_thread

        boards = tolist(boards)
        roots = self.boards2roots
        nodes = {board: boards2roots.setdefault(board, Node()) for board in boards}
        containers = {board: [] for board in boards}
        counts = {board: rollout_time for board in boards}
        thread_containers = {board: [] for board in boards}
        thread_counts = {board: rollout_time for board in boards}
        actions = {}
        lock = threading.RLock()
        while counts:
            for board, thread_container in thread_containers.iteritems():
                root = boards2roots[board]
                container = containers[board]
                while len(thread_container) < max_thread and thread_counts[board]:
                    search_thread = SearchThread(root, board.copy(), lock, container)
                    thread_container.append(search_thread)
                    thread_counts[board] -= 1
                    search_thread.run()
            for board, container in containers.iteritems():
                if thread_counts[board] < evaluation_size:
                #TO DO

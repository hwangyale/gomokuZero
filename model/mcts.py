from __future__ import print_function

import threading
import warnings
import time

import numpy as np
from ..constant import *
from ..utils import tolist, sample

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
        name = self.getName()

        node = self.root
        board = self.board
        lock = self.lock

        lock.acquire()
        # print(id(self.container))
        count = 0
        if not node.children:
            self.container.append((node, board, self))
            lock.release()
            # print('{}: {}'.format(name, count))
        else:
            lock.release()
            while not board.is_over:
                lock.acquire()
                position, node = node.select()
                count += 1
                # print('{}: {} {}'.format(name, position, count))
                # print('child nodes: {}'.format(node.children))
                if not node.children:
                    board.move(position)
                    self.container.append((node, board, self))
                    lock.release()
                    break
                lock.release()
                board.move(position)
        # print(len(self.container))


class MCTS(object):
    def __init__(self, policyValueModel, rollout_time=100, max_thread=8, evaluation_size=8):
        self.policyValueModel = policyValueModel
        if max_thread < evaluation_size:
            warnings.warn('`max_thread` is less than `evaluation_size`, '
                          'setting: `evaluation_size = max_thread`')
            evaluation_size = max_thread
        self.rollout_time = rollout_time
        self.max_thread = max_thread
        self.evaluation_size = evaluation_size

        self.boards2roots = {}

    def rollout(self, boards, Tau=1.0,
                rollout_time=None, max_thread=None,
                evaluation_size=None):
        if Tau < 0.0:
            raise Exception('Tau < 0.0')

        if rollout_time is None:
            rollout_time = self.rollout_time
        if max_thread is None:
            max_thread = self.max_thread
        if evaluation_size is None:
            evaluation_size = self.evaluation_size
        if max_thread < evaluation_size:
            warnings.warn('`max_thread` is less than `evaluation_size`, '
                          'setting: `evaluation_size = max_thread`')
            evaluation_size = max_thread

        boards = tolist(boards)
        roots = {board: self.boards2roots.setdefault(board, Node()) for board in boards}
        containers = {board: [] for board in boards}
        counts = {board: rollout_time for board in boards}
        thread_containers = {board: set() for board in boards}
        thread_counts = {board: rollout_time for board in boards}
        positions = {}
        lock = threading.RLock()
        while counts:
            # print('test', len(containers[boards[0]]))
            # time.sleep(1)
            lock.acquire()
            cache_threads = []
            for board, thread_container in thread_containers.iteritems():
                root = roots[board]
                container = containers[board]
                while len(thread_container) < min(max_thread, thread_counts[board]) \
                        and len(container) < min(max_thread, thread_counts[board]):
                    # print('`thread_container`: {}, {}'.format(len(thread_container), thread_counts[board]))
                    # print('`container`: {}, {}'.format(len(container), thread_counts[board]))
                    search_thread = SearchThread(root, board.copy(), lock, container)
                    thread_container.add(search_thread)
                    thread_counts[board] -= 1
                    cache_threads.append(search_thread)

            for search_thread in cache_threads:
                search_thread.run()

            if not all([
                len(container) >= min(evaluation_size, counts[board]) \
                for board, container in containers.iteritems()
            ]):
                # print('test test test 1')
                # time.sleep(MAIN_THREAD_SLEEP_TIME)
                lock.release()
                continue
            else:
                # print('test test test 2')
                pass

            backup_nodes = []
            hashing_boards = []
            evaluation_boards = []
            finished_threads = []
            for board, container in containers.iteritems():
                while container:
                    _node, _board, _thread = container.pop()
                    backup_nodes.append(_node)
                    hashing_boards.append(board)
                    evaluation_boards.append(_board)
                    finished_threads.append(_thread)
                    counts[board] -= 1

            policies, values = self.policyValueModel.get_policy_values(evaluation_boards, True)
            policies = tolist(policies)

            for idx, node in enumerate(backup_nodes):
                evaluation_board = evaluation_boards[idx]
                if evaluation_board.is_over:
                    winner = evaluation_board.winner
                    if winner == DRAW:
                        value = 0.0
                    else:
                        value = -1.0
                else:
                    if len(node.children) == 0:
                        # print(node.parent is None, node.parent == roots[hashing_boards[idx]])
                        policy = policies[idx]
                        node.expand(policy)
                    value = values[idx]
                node.backup(-value)

                hashing_board = hashing_boards[idx]
                finished_thread = finished_threads[idx]
                thread_containers[hashing_board].remove(finished_thread)

            finished_boards = []
            for board, count in counts.iteritems():
                if count == 0:
                    finished_boards.append(board)
                    del thread_containers[board]
                    del thread_counts[board]
                    del containers[board]

            for board in finished_boards:
                root = roots[board]
                if Tau == 0.0:
                    position = max(root.children.items(), key=lambda (a, n): n.N)[0]
                else:
                    legal_positions = root.children.keys()
                    ps = np.array([n.N**(1/Tau) for n in root.children.values()])
                    ps /= np.sum(ps)
                    position = legal_positions[sample(ps)]
                positions[board] = position

                del counts[board]

            # print(counts)
            lock.release()

        return positions

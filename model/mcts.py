import threading
import warnings
import time

import numpy as np
from ..constant import *
from ..utils import tolist, sample
from .neural_network import PolicyValueNetwork
from ..board.board import Board

class Node(object):
    def __init__(self, prior=1.0, parent=None, children=None):
        self.parent = parent
        if children is None:
            self.children = {}
        else:
            self.children = children

        self.P = prior
        self.N = 0.0
        self.W = 0.0
        self.is_virtual = []

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
            node.is_virtual.append(None)
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

        while self.is_virtual:
            self.W += VIRTUAL_LOSS
            self.N -= VIRTUAL_VISIT
            self.is_virtual.pop()

        if self.parent:
            self.parent.backup(-value)

    def get_config(self):
        config = {}
        config['P'] = self.P
        config['N'] = self.N
        config['W'] = self.W
        config['is_virtual'] = [i_v for i_v in self.is_virtual]
        config['children'] = {
            position: child_node.get_config() \
            for position, child_node in self.children.iteritems()
        }

        return config

    @classmethod
    def instantiate_from_config(cls, config):
        node = cls()
        children_config = config.pop('children')
        node.__dict__.update(config)
        for position, child_config in children_config.iteritems():
            child_node = cls.instantiate_from_config(child_config)
            child_node.parent = node
            node.children[position] = child_node
        return node


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
            self.container.append((node, board, self))
            lock.release()
        else:
            lock.release()
            while not board.is_over:
                lock.acquire()
                position, node = node.select()
                if not node.children:
                    board.move(position)
                    self.container.append((node, board, self))
                    break
                lock.release()
                board.move(position)
            lock.release()


class MCTS(object):
    def __init__(self, policyValueModel, rollout_time=100, max_thread=8):
        self.policyValueModel = policyValueModel
        self.rollout_time = rollout_time
        self.max_thread = max_thread
        self.boards2roots = {}
        self.boards2policies = {}

    def rollout(self, boards, Tau=1.0,
                rollout_time=None, max_thread=None):
        if Tau < 0.0:
            raise Exception('Tau < 0.0')

        if rollout_time is None:
            rollout_time = self.rollout_time
        if max_thread is None:
            max_thread = self.max_thread

        boards = tolist(boards)
        roots = {board: self.boards2roots.setdefault(board, Node()) for board in boards}
        containers = {board: [] for board in boards}
        counts = {board: rollout_time for board in boards}
        thread_containers = {board: set() for board in boards}
        thread_counts = {board: rollout_time for board in boards}
        boards2policies = {}
        lock = threading.RLock()

        backup_count = 0
        while counts:
            cache_threads = []
            for board, thread_container in thread_containers.iteritems():
                root = roots[board]
                container = containers[board]
                while len(thread_container) < max_thread and thread_counts[board]:
                    search_thread = SearchThread(root, board.copy(), lock, container)
                    thread_container.add(search_thread)
                    thread_counts[board] -= 1
                    search_thread.start()
                    time.sleep(PROCESS_SLEEP_TIME)
                    cache_threads.append(search_thread)

            lock.acquire()

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

            if len(evaluation_boards) == 0:
                lock.release()
                continue

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
                legal_positions = root.children.keys()
                root_children = root.children.values()
                if Tau == 0.0:
                    policy = {l_p: 0.0 for l_p in legal_positions}
                    position = max(zip(legal_positions, root_children),
                                   key=lambda (a, n): n.N)[0]
                    policy[position] = 1.0
                else:
                    ps = np.array([n.N**(1/Tau) for n in root_children])
                    ps = (ps / np.sum(ps)).tolist()
                    policy = {l_p: ps[i] for i, l_p in enumerate(legal_positions)}
                boards2policies[board] = policy

                del counts[board]

            lock.release()

        return boards2policies

    def get_policies(self, boards, Tau=1.0,
                     rollout_time=None, max_thread=None):
        boards = tolist(boards)
        boards2roots = self.boards2roots
        for board in boards:
            if board not in boards2roots:
                continue
            root = boards2roots[board]
            step = root.step
            node = root
            for position in board.history[step:]:
                if position in node.children:
                    node = node.children[position]
                else:
                    node = Node()
                    break
            node.step = len(board.history)
            boards2roots[board] = node
        policies = self.rollout(boards, Tau=Tau,
                                rollout_time=rollout_time,
                                max_thread=max_thread)
        self.boards2policies.update(policies)

        if len(boards) == 1:
            return policies[boards[0]]
        else:
            return [policies[board] for board in boards]

    def get_positions(self, boards, Tau=1.0,
                      rollout_time=None, max_thread=None):
        boards = tolist(boards)
        boards2positions = {}
        rollout_boards = []
        for board in boards:
            policy = self.boards2policies.pop(board, None)
            if policy is None:
                rollout_boards.append(board)
                continue
            boards2positions[board] = sample(policy)

        if len(rollout_boards):
            self.get_policies(rollout_boards, Tau=Tau,
                              rollout_time=rollout_time, max_thread=max_thread)
            for board in rollout_boards:
                boards2positions[board] = sample(self.boards2policies.pop(board))

        if len(boards) == 1:
            return boards2positions[boards[0]]
        else:
            return [boards2positions[board] for board in boards]

    def get_config(self, path=None, weights_path=None):
        config = {}
        config['rollout_time'] = self.rollout_time
        config['max_thread'] = self.max_thread
        boards = []
        roots = []
        policies = []
        for board, root in self.boards2roots.iteritems():
            boards.append(board.get_config())
            roots.append(root.get_config())
            policies.append(self.boards2policies.get(board, None))
        config['boards'] = boards
        config['roots'] = roots
        config['policies'] = policies

        if path is not None:
            self.policyValueModel.save_model(path, weights_path)
            config['path'] = path

        return config

    @classmethod
    def instantiate_from_config(cls, config):
        path = config.get('path', None)
        if path is None:
            warnings.warn('Unknown `policyValueModel`')
            policyValueModel = None
        else:
            policyValueModel = PolicyValueNetwork.load_model(path)
        mcts = cls(policyValueModel, config['rollout_time'], config['max_thread'])
        boards = config['boards']
        roots = config['roots']
        policies = config['policies']
        for idx, board_config in enumerate(boards):
            board = Board.instantiate_from_config(board_config)
            root = Node.instantiate_from_config(roots[idx])
            mcts.boards2roots[board] = root
            if policies[idx] is not None:
                mcts.boards2policies[board] = policies[idx]

        return mcts

import multiprocessing
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
        '''lock the process
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
        '''lock the process
        '''
        if self.children:
            raise Exception('Expand the expanded node')
        for l_p, prob in policy.iteritems():
            self.children[l_p] = self.__class__(prob, self)

    def backup(self, value):
        '''lock the process
        '''
        self.W += value
        self.N += 1.0
        while self.is_virtual:
            self.W += VIRTUAL_LOSS
            self.N -= VIRTUAL_VISIT
            self.is_virtual.pop()
        if self.parent:
            self.parent.backup(-value)


class SearchProcess(multiprocessing.Process):
    def __init__(self, root, board, lock, container, name=None):
        self.root = root
        self.board = board
        self.lock = lock
        self.container = container
        super(SearchProcess, self).__init__(name=name)

    def run(self):
        node = self.root
        board = self.board
        lock = self.lock

        lock.acquire()
        count = 0
        if not node.children:
            self.container.append((node, board, self))
            lock.release()
        else:
            lock.release()
            while not board.is_over:
                lock.acquire()
                count += 1
                position, node = node.select()
                if not node.children:
                    board.move(position)
                    self.container.append((node, board, self))
                    lock.release()
                    break
                lock.release()
                board.move(position)


class MCTS(object):
    def __init__(self, policyValueModel, rollout_time=100, max_process=8):
        self.policyValueModel = policyValueModel
        self.rollout_time = rollout_time
        self.max_process = max_process
        self.boards2roots = {}
        self.boards2policies = {}

    def rollout(self, boards, Tau=1.0,
                rollout_time=None, max_process=None):
        if Tau < 0.0:
            raise Exception('Tau < 0.0')

        if rollout_time is None:
            rollout_time = self.rollout_time
        if max_process is None:
            max_process = self.max_process

        boards = tolist(boards)
        roots = {board: self.boards2roots.setdefault(board, Node()) for board in boards}
        containers = {board: [] for board in boards}
        counts = {board: rollout_time for board in boards}
        process_containers = {board: set() for board in boards}
        process_counts = {board: rollout_time for board in boards}
        boards2policies = {}
        lock = multiprocessing.RLock()
        while counts:
            for board, process_container in process_containers.iteritems():
                root = roots[board]
                container = containers[board]
                while len(process_container) < min(max_process, process_counts[board]):
                    search_process = SearchProcess(root, board.copy(), lock, container)
                    process_container.add(search_process)
                    process_counts[board] -= 1
                    search_process.daemon = True
                    search_process.run()
                    time.sleep(PROCESS_SLEEP_TIME)

            lock.acquire()

            backup_nodes = []
            hashing_boards = []
            evaluation_boards = []
            finished_processes = []
            for board, container in containers.iteritems():
                while container:
                    _node, _board, _process = container.pop()
                    backup_nodes.append(_node)
                    hashing_boards.append(board)
                    evaluation_boards.append(_board)
                    finished_processes.append(_process)
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
                finished_process = finished_processes[idx]
                process_containers[hashing_board].remove(finished_process)

            finished_boards = []
            for board, count in counts.iteritems():
                if count == 0:
                    finished_boards.append(board)
                    del process_containers[board]
                    del process_counts[board]
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
                    ps /= np.sum(ps)
                    policy = {l_p: ps[i] for i, l_p in enumerate(legal_positions)}
                boards2policies[board] = policy

                del counts[board]

            lock.release()

        return boards2policies

    def get_policies(self, boards, Tau=1.0,
                     rollout_time=None, max_process=None):
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
                                max_process=max_process)
        self.boards2policies.update(policies)

        if len(boards) == 1:
            return policies[boards[0]]
        else:
            return [policies[board] for board in boards]

    def get_positions(self, boards, Tau=1.0,
                      rollout_time=None, max_process=None):
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
                              rollout_time=rollout_time, max_process=max_process)
            for board in rollout_boards:
                boards2positions[board] = sample(self.boards2policies.pop(board))

        if len(boards) == 1:
            return boards2positions[boards[0]]
        else:
            return [boards2positions[board] for board in boards]

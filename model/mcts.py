from __future__ import print_function

import sys
import threading
import warnings
import time

import numpy as np
from ..constant import *
from ..utils import tolist, sample
from .neural_network import PolicyValueNetwork
from ..board.board import Board
from ..utils.progress_bar import ProgressBar

class Node(object):
    def __init__(self, prior=1.0, parent=None, children=None,
                 step=None):
        self.parent = parent
        if children is None:
            self.children = {}
        else:
            self.children = children

        self.P = prior
        self.N = 0.0
        self.W = 0.0
        self.is_virtual = []
        self.step = step

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
        if self.step is not None:
            step = self.step + 1
        else:
            step = None
        for l_p, prob in policy.iteritems():
            self.children[l_p] = self.__class__(prob, self, step=step)

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
            while True:
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
    def __init__(self, policyValueModel, rollout_time=100, max_thread=8,
                 exploration_epsilon=0.0):
        self.policyValueModel = policyValueModel
        self.rollout_time = rollout_time
        self.max_thread = max_thread
        self.exploration_epsilon = exploration_epsilon
        self.boards2roots = {}
        self.boards2policies = {}

    def rollout(self, boards, Tau=1.0,
                rollout_time=None, max_thread=None,
                exploration_epsilon=None, verbose=0):
        boards = tolist(boards)
        if not isinstance(Tau, list):
            Tau = [Tau] * len(boards)
        else:
            assert len(Tau) == len(boards)
        if any([t < 0.0 for t in Tau]):
            raise Exception('Tau < 0.0')
        boards2Taus = {}
        for idx, board in enumerate(boards):
            boards2Taus[board] = Tau[idx]

        if rollout_time is None:
            rollout_time = self.rollout_time
        if max_thread is None:
            max_thread = self.max_thread
        if exploration_epsilon is None:
            exploration_epsilon = [self.exploration_epsilon] * len(boards)
        elif isinstance(exploration_epsilon, list):
            assert len(exploration_epsilon) == len(boards)
        else:
            exploration_epsilon = [exploration_epsilon] * len(boards)

        roots = {board: self.boards2roots.setdefault(board, Node(step=len(board.history)))
                 for board in boards}

        #add noise to prior probabilities of child nodes of roots
        if exploration_epsilon:
            boards2epsilons = {}
            for idx, board in enumerate(boards):
                root = roots[board]
                epsilon = exploration_epsilon[idx]
                if not root.children:
                    boards2epsilons[board] = epsilon
                    continue
                alphas = (DIRICHLET_ALPHA, ) * len(root.children)
                dirichlet_noises = np.random.dirichlet(alphas).tolist()
                for child_node in root.children.values():
                    child_node.P = (1 - epsilon) * child_node.P + epsilon * dirichlet_noises.pop()

        containers = {board: [] for board in boards}
        counts = {board: rollout_time for board in boards}
        thread_containers = {board: set() for board in boards}
        thread_counts = {board: rollout_time for board in boards}
        boards2policies = {}
        lock = threading.RLock()

        if verbose:
            total_step = len(boards) * rollout_time
            progress_bar = ProgressBar(total_step)
            current_step = 0
        while counts:
            for board, thread_container in thread_containers.iteritems():
                root = roots[board]
                container = containers[board]
                while len(thread_container) < max_thread and thread_counts[board]:
                    search_thread = SearchThread(root, board.copy(), lock, container)
                    thread_container.add(search_thread)
                    thread_counts[board] -= 1
                    search_thread.start()
                    time.sleep(PROCESS_SLEEP_TIME)

            lock.acquire()

            backup_nodes = []
            hashing_boards = []
            evaluation_boards = []
            finished_threads = []
            for board, container in containers.iteritems():
                while container:
                    counts[board] -= 1
                    _node, _board, _thread = container.pop()
                    if _board.is_over:
                        if _board.winner:
                            _node.backup(0.0)
                        else:
                            _node.backup(1.0)
                        thread_containers[board].remove(_thread)
                    else:
                        backup_nodes.append(_node)
                        hashing_boards.append(board)
                        evaluation_boards.append(_board)
                        finished_threads.append(_thread)
                    if verbose:
                        current_step += 1

            if len(backup_nodes):

                policies, values = self.policyValueModel.get_policy_values(evaluation_boards, True)
                policies = tolist(policies)

                for idx, node in enumerate(backup_nodes):
                    if len(node.children) == 0:
                        policy = policies[idx]
                        if exploration_epsilon and node.parent is None:
                            alphas = (DIRICHLET_ALPHA, ) * len(policy)
                            dirichlet_noises = np.random.dirichlet(alphas).tolist()
                            epsilon = boards2epsilons[hashing_boards[idx]]
                            for l_p, prob in policy.iteritems():
                                policy[l_p] = (1 - epsilon) * prob + epsilon * dirichlet_noises.pop()

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
                tau = boards2Taus[board]
                if tau == 0.0:
                    position = max(zip(legal_positions, root_children),
                                   key=lambda (a, n): n.N)[0]
                    policy = {position: 1.0}
                else:
                    ps = np.array([n.N**(1/tau) for n in root_children])
                    ps = (ps / np.sum(ps)).tolist()
                    policy = {l_p: ps[i] for i, l_p in enumerate(legal_positions)}
                boards2policies[board] = policy

                del counts[board]

            if verbose:
                progress_bar.update(current_step)
            lock.release()
        if verbose:
            sys.stdout.write(' '*79 + '\r')
            sys.stdout.flush()

        return boards2policies

    def get_policies(self, boards, Tau=1.0,
                     rollout_time=None, max_thread=None,
                     exploration_epsilon=0.0, verbose=0):
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
                    node = Node(step=len(board.history))
                    break
            node.parent = None
            boards2roots[board] = node
        policies = self.rollout(boards, Tau=Tau,
                                rollout_time=rollout_time,
                                max_thread=max_thread,
                                exploration_epsilon=exploration_epsilon,
                                verbose=verbose)
        self.boards2policies.update(policies)

        if len(boards) == 1:
            return policies[boards[0]]
        else:
            return [policies[board] for board in boards]

    def get_positions(self, boards, Tau=1.0,
                      rollout_time=None, max_thread=None,
                      exploration_epsilon=0.0, verbose=0):
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
                              rollout_time=rollout_time, max_thread=max_thread,
                              exploration_epsilon=exploration_epsilon,
                              verbose=verbose)
            for board in rollout_boards:
                boards2positions[board] = sample(self.boards2policies.pop(board))

        if len(boards) == 1:
            if verbose == 2:
                board = boards[0]
                position = boards2positions[board]
                root = self.boards2roots[board]
                child_node = root.children[position]
                print('prior prob:{:.4f} visit times:{:d} Q value:{:.4f}'
                      .format(child_node.P, int(child_node.N), child_node.Q))
                return position
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

    def clear(self, boards=None):
        if boards is None:
            self.boards2roots = {}
            self.boards2policies = {}
            return None
        for board in boards:
            self.boards2roots.pop(board, None)
            self.boards2policies.pop(board, None)

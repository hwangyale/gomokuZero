import os
import time
from gomokuZero.board.play import *
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS
from gomokuZero.board.board import Board
from gomokuZero.constant import *
from gomokuZero.utils.board_utils import get_hashing_key_of_board
from gomokuZero.utils.vct import get_vct, HASHING_TABLE_OF_VCT


try:
    input = raw_input
except NameError:
    pass

class TestGame(Game):
    def play(self, board=None, Tau=0.0, verbose=2):
        if board is None:
            board = Board()

        boards2nodes = dict()
        def hash_nodes(_root):
            node_stack = [_root]
            sign = {BLACK: 1, WHITE: -1}[_root.board.player]
            while len(node_stack):
                node = node_stack.pop()
                _board = node.board
                _board_key = get_hashing_key_of_board(_board)
                boards2nodes[sign*_board_key] = node
                for child_node in node.children.values():
                    node_stack.append(child_node)

        def print_vct_node(_board):
            _board_key = get_hashing_key_of_board(_board)
            sign = {BLACK: 1, WHITE: -1}[_board.player]
            _root = boards2nodes.get(sign*_board_key, None)

            node_information = {BLACK: 'black:\n', WHITE: 'white:\n'}[_board.player]
            if _root is None:
                return None
            elif _root.selected_node is None:
                return node_information + 'No root move'

            root_child = _root.selected_node[1]
            node_information += 'root move:{} proof:{:d} disproof:{:d}\n'.format(
                position2action(_root.selected_node[0]),
                root_child.proof, root_child.disproof
            )
            if len(root_child.children):
                node_information += 'responded move:\n'
                for position, node in root_child.children.items():
                    node_information += 'move:{} proof:{:d} disproof:{:d} children:{}\n'.format(
                        position2action(position),
                        node.proof, node.disproof,
                        list(map(position2action, list(node.children.keys())))
                    )
            else:
                node_information += 'no responded move and the board key is '
                copy_board = _board.copy()
                copy_board.move(_root.selected_node[0])
                copy_board_key = sign * get_hashing_key_of_board(copy_board)
                if copy_board_key in HASHING_TABLE_OF_VCT:
                    node_information += 'in '
                else:
                    node_information += 'not in '
                node_information += 'the hashing table of vct\n'

            return node_information

        time_delay = self.time_delay
        os.system('cls')
        print(board)
        print('\n')
        while not board.is_over:
            player = {BLACK: self.black_player,
                      WHITE: self.white_player}[board.player]
            root_container = []
            if hasattr(player, 'ai') and isinstance(player.ai, MCTS):
                position = player.get_position(board, Tau=Tau, verbose=verbose, root_container=root_container)
            else:
                get_vct(board, ROOT_MAX_DEPTH, ROOT_MAX_TIME, locked=True, root_container=root_container)
                position = player.get_position(board)

            if len(root_container):
                hash_nodes(root_container[0])

            node_information = print_vct_node(board)

            board.move(position)
            time.sleep(time_delay)
            os.system('cls')
            print(board)
            print('move: ({:d}, {:d})'.format(position[0]+1, position[1]+1))

            if node_information is not None:
                print(node_information)
                time.sleep(5)
            # if hasattr(player, 'ai') and isinstance(player.ai, MCTS) and len(root_container):
            #     root = root_container[0]
            #     root_child = root.selected_node[1]
            #     node_information = 'root move:{} proof:{:d} disproof:{:d}\nresponded move:\n'.format(
            #         root.selected_node[0], root_child.proof, root_child.disproof
            #     )
            #     for position, node in root_child.children.items():
            #         node_information += 'move:{} proof:{:d} disproof:{:d} children:{}\n'.format(
            #             position, node.proof, node.disproof, list(node.children.keys())
            #         )
            #     print(node_information)
            print('\n')
        os.system('cls')
        print(board)
        print('\n')
        result = '\n'
        if board.winner == DRAW:
            result += 'draw'
        else:
            if board.winner == BLACK:
                result += 'black wins'
            else:
                result += 'white wins'

            start = position2action(board.five[0])
            end = position2action(board.five[1])
            result += ' from {} to {}'.format(start, end)
        print(result)


def get_game(nn_path, mcts_config, time_delay):
    player = Player()
    mcts = AIPlayer(MCTS(
        PolicyValueNetwork.load_model(nn_path), **mcts_config
    ))

    while True:
        os.system('cls')
        color = input('black: 1\nwhite: 2\n')
        try:
            color = int(color)
            if color in [1, 2]:
                break
        except:
            pass

    if color == 1:
        # game = Game(player, mcts, time_delay=time_delay)
        game = TestGame(player, mcts, time_delay=time_delay)
    else:
        # game = Game(mcts, player, time_delay=time_delay)
        game = TestGame(mcts, player, time_delay=time_delay)

    # _play = game.play
    # def play():
    #     _play(Tau=0.0, verbose=2)
    #
    # game.play = play

    return game



if __name__ == '__main__':
    import keras.backend as K
    if K.backend() == 'theano':
        nn_path = 'data/pre_train/yixin_version_nn_config.json'
    else:
        # nn_path = 'data/pre_train/yixin_version_tf_nn_config.json'
        nn_path = 'data/pre_train/tournament_version_tf_nn_config.json'
    mcts_config = {
        'rollout_time': 100, 'max_thread': 1, 'gamma': 0.0, 'max_depth': 6
    }

    game = get_game(nn_path, mcts_config, 2)
    game.play()

import os
from gomokuZero.board.play import AIPlayer, Game, Player
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS

try:
    input = raw_input
except NameError:
    pass


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
        game = Game(player, mcts, time_delay=time_delay)
    else:
        game = Game(mcts, player, time_delay=time_delay)

    _play = game.play
    def play():
        _play(Tau=0.0, verbose=2)

    game.play = play

    return game



if __name__ == '__main__':
    nn_path = 'data/pre_train/yixin_version_nn_config.json'
    mcts_config = {
        'rollout_time': 100, 'max_thread': 1, 'gamma': 0.0, 'max_depth': 6
    }

    game = get_game(nn_path, mcts_config, 2)
    game.play()

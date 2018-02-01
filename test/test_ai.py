from gomokuZero.board.play import AIPlayer, Game
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS

# nn_path = '/data/zero/test_version_nn_config.json'
# nn_path = 'data/pre_train/pre_train_version_nn_config.json'
# nn_path = 'data/pre_train/yixin_version_nn_config.json'
nn_path = 'data/pre_train/input_coding_version_nn_config.json'
# nn_path = 'data/cache/cache_pre_train_version_nn_config.json'
mcts_config = {
    'rollout_time': 100, 'max_thread': 8, 'gamma': 0.0
}

black_player = AIPlayer(MCTS(
    PolicyValueNetwork.load_model(nn_path), **mcts_config
))
# white_player = AIPlayer(MCTS(
    # PolicyValueNetwork.load_model(nn_path), **mcts_config
# ))
white_player = AIPlayer(
    PolicyValueNetwork.load_model(nn_path)
)

game = Game(black_player, white_player, time_delay=2)
game.play(Tau=0.0, verbose=2)

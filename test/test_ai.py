from gomokuZero.board.play import AIPlayer, Game, Tournament
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS

# nn_path = '/data/zero/test_version_nn_config.json'
# nn_path = 'data/pre_train/pre_train_version_nn_config.json'
nn_path = 'data/pre_train/yixin_version_nn_config.json'
# nn_path = 'data/pre_train/input_coding_version_nn_config.json'
# nn_path = 'data/pre_train/input_coding_augmentation_version_nn_config.json'
# nn_path = 'data/cache/cache_pre_train_version_nn_config.json'
mcts_config = {
    'rollout_time': 100, 'max_thread': 1, 'gamma': 0.0, 'max_depth': 4
}

case = 1
if case == 1:
    player_1 = MCTS(
        PolicyValueNetwork.load_model(nn_path), **mcts_config
    )
    player_2 = PolicyValueNetwork.load_model(nn_path)

    tournament = Tournament(player_1, player_2)
    tournament.play(10, 5)

elif case == 2:
    black_player = AIPlayer(MCTS(
        PolicyValueNetwork.load_model(nn_path), **mcts_config
    ))
    # white_player = AIPlayer(MCTS(
    #     PolicyValueNetwork.load_model(nn_path), **mcts_config
    # ))
    white_player = AIPlayer(
        PolicyValueNetwork.load_model(nn_path)
    )

    game = Game(black_player, white_player, time_delay=2)
    game.play(Tau=0.0, verbose=2)

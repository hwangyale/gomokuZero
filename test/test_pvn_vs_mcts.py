from __future__ import print_function

import sys
import numpy as np
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.model.mcts import MCTS
from gomokuZero.constant import *
from gomokuZero.board.board import Board
from gomokuZero.utils import tolist
from gomokuZero.utils.progress_bar import ProgressBar

def fight(nn_path, game_number, batch_size, rollout_time, max_thread):
    pvn = PolicyValueNetwork.load_model(nn_path)
    mcts = MCTS(
        PolicyValueNetwork.load_model(nn_path),
        rollout_time=rollout_time, max_thread=max_thread
    )
    wins = {pvn: 0, mcts: 0}
    progress_bar = ProgressBar(game_number*2)
    progress_bar.update(0)
    for player, opponent, step_count in [(pvn, mcts, 0), (mcts, pvn, game_number)]:
        player.color = BLACK
        opponent.color = WHITE
        boards = set()
        game_count = game_number
        while boards or game_count:
            for _ in range(min(batch_size-len(boards), game_count)):
                boards.add(Board())
            cache_boards = list(boards)
            if len(cache_boards) == 0:
                print('No boards to play')
                continue
            positions = player.get_positions(cache_boards, Tau=1.0)
            positions = tolist(positions)
            finished_boards = []
            for board, position in zip(cache_boards, positions):
                board.move(position)
                if board.is_over:
                    winner = board.winner
                    if winner == pvn.color:
                        wins[pvn] += 1
                    elif winner == mcts.color:
                        wins[mcts] += 1
                    finished_boards.append(board)

            sys.stdout.write(
                ' '*70 + '{:d}\r'.format(
                    int(np.mean([len(board.history) for board in boards]))
                )
            )
            sys.stdout.flush()

            for board in finished_boards:
                game_count -= 1
                boards.remove(board)

            player, opponent = opponent, player
            progress_bar.update(game_number-game_count+step_count)
    sys.stdout.write(' '*79 + '\r')
    sys.stdout.flush()

    print('pvn wins:{:.2f}'.format((wins[pvn]+0.0)/game_number/2*100))
    print('mcts wins:{:.2f}'.format((wins[mcts]+0.0)/game_number/2*100))

if __name__ == '__main__':
    nn_path = '/data/zero/test_version_nn_config.json'
    fight(
        nn_path=nn_path, game_number=25,
        batch_size=5, rollout_time=256, max_thread=6
    )

from __future__ import print_function
from gomokuZero.train.pipeline import get_samples
from gomokuZero.model.neural_network import PolicyValueNetwork

pvn = PolicyValueNetwork(blocks=3, filters=16)
board_tensors, policy_tensors, value_tensors = get_samples(
    pvn, 10, 5, game_batch_size=4, max_thread=4, rollout_time=100
)
print(board_tensors.shape, policy_tensors.shape, value_tensors.shape)

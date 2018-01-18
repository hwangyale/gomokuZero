from __future__ import print_function
from gomokuZero.train.pipeline import get_samples, Trainer
from gomokuZero.model.neural_network import PolicyValueNetwork

pvn = PolicyValueNetwork(blocks=3, filters=16)
# board_tensors, policy_tensors, value_tensors = get_samples(
#     pvn, 10, 5, game_batch_size=2, max_thread=4, rollout_time=100
# )
# print(board_tensors.shape, policy_tensors.shape, value_tensors.shape)
# trainer = Trainer(
#     pvn, **{
#         'blocks': 3,
#         'kernel_size': (3, 3),
#         'filters': 16,
#         'game_number': 10,
#         'step_to_explore': 3,
#         'game_batch_size': 4,
#         'augment': True,
#         'rollout_time': 32,
#         'max_thread': 4,
#         'epochs': 4,
#         'train_steps': 128,
#         'train_batch_size': 16,
#         'lr': {
#             400: 1e-2,
#             600: 1e-3,
#             float('inf'): 1e-4
#         },
#         'momentum': 0.9,
#         'value_loss_weight': 1.0,
#         'weight_decay': 1e-4
#     }
# )
trainer = Trainer.load('test.json')
trainer.epochs = 8
trainer.setting['epochs'] = 8
trainer.run('test.json', 'test_pvn.json', 'test_pvn.npz', 'data.npz')

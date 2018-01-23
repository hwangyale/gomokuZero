from __future__ import print_function
from gomokuZero.utils import check_load_path
from gomokuZero.train.pipeline import Trainer
from gomokuZero.model.neural_network import PolicyValueNetwork

pvn = PolicyValueNetwork(blocks=3, filters=16)
# board_tensors, policy_tensors, value_tensors = get_samples(
#     pvn, 10, 5, game_batch_size=2, max_thread=4, rollout_time=100
# )
# print(board_tensors.shape, policy_tensors.shape, value_tensors.shape)
if check_load_path('test.json') is None:
    trainer = Trainer(
        pvn, **{
            'blocks': 2,
            'kernel_size': (3, 3),
            'filters': 16,
            'game_number': 8,
            'step_to_explore': 3,
            'game_batch_size': 4,
            'pool_size': 40,
            'pool_point': 0,
            'augment': False,
            'rollout_time': 32,
            'max_thread': 16,
            'epochs': 4,
            'train_batch_size': 128,
            'train_epochs': 2,
            'lr': {
                400: 1e-2,
                600: 1e-3,
                float('inf'): 1e-4
            },
            'momentum': 0.9,
            'value_loss_weight': 1.0,
            'weight_decay': 1e-4
        }
    )
else:
    trainer = Trainer.load('test.json')
# trainer.epochs = 8
# trainer.setting['epochs'] = 8
trainer.run('test.json', 'test_pvn.json', 'test_pvn.npz', 'data.npz')

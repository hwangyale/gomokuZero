from __future__ import print_function

import json
import numpy as np
from gomokuZero.constant import *
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.utils import check_load_path

try:
    range = xrange
except NameError:
    pass

nn_path = 'data/pre_train/yixin_version_nn_config.json'
pvn = PolicyValueNetwork.load_model(nn_path)
# pvn = PolicyValueNetwork()

samples = np.load(check_load_path('data/records/yixin_samples.npz'))
board_tensors = samples['board_tensors']
policy_tensors = samples['policy_tensors'].reshape((-1, SIZE**2))

try:
    trainer_path = 'data/cache/cache_pre_trainer.json'
    with open(check_load_path(trainer_path), 'r') as f:
        trainer_config = json.load(f)
    train_idxs = trainer_config['train_idxs']
    idxs = list(range(board_tensors.shape[0]))
    test_idxs = list(set(idxs) - set(train_idxs))
    board_test = board_tensors[test_idxs, ...]
    policy_test = policy_tensors[test_idxs, ...]
except:
    board_test = board_tensors
    policy_test = policy_tensors

pvn.policy_model.compile(loss='categorical_crossentropy', optimizer='SGD',
                         metrics=['acc'])
loss, acc = pvn.policy_model.evaluate(board_test, policy_test)
print('accuracy:{:.2f}%'.format(acc*100))

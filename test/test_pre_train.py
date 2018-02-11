from __future__ import print_function

import json
import numpy as np
import keras.backend as K
from gomokuZero.constant import *
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.utils import check_load_path

try:
    range = xrange
except NameError:
    pass

if K.backend() == 'theano':
    nn_path = 'data/pre_train/yixin_version_nn_config.json'
    # nn_path = 'data/pre_train/input_coding_version_nn_config.json'
    # nn_path = 'data/pre_train/input_coding_augmentation_version_nn_config.json'
else:
    nn_path = 'data/pre_train/yixin_version_tf_nn_config.json'
pvn = PolicyValueNetwork.load_model(nn_path)
# pvn = PolicyValueNetwork()

samples = np.load(check_load_path('data/records/yixin_samples.npz'))
board_tensors = samples['board_tensors']
if K.backend() == 'tensorflow':
    board_tensors = np.transpose(board_tensors, (0, 2, 3, 1))
policy_tensors = samples['policy_tensors'].reshape((-1, SIZE**2))
value_tensors = samples['value_tensors'][:, :1]

try:
    trainer_path = 'data/cache/cache_yixin_version_pre_trainer.json'
    with open(check_load_path(trainer_path), 'r') as f:
        trainer_config = json.load(f)
    train_idxs = trainer_config['train_idxs']
    idxs = list(range(board_tensors.shape[0]))
    test_idxs = list(set(idxs) - set(train_idxs))
    board_test = board_tensors[test_idxs, ...]
    policy_test = policy_tensors[test_idxs, ...]
    value_test = value_tensors[test_idxs, ...]
except:
    board_test = board_tensors
    policy_test = policy_tensors
    value_test = value_tensors

pvn.policy_model.compile(loss='categorical_crossentropy', optimizer='SGD',
                         metrics=['acc'])
loss, acc = pvn.policy_model.evaluate(board_test, policy_test, batch_size=128)
print('accuracy:{:.2f}%'.format(acc*100))

pvn.value_model.compile(loss='mean_squared_error', optimizer='SGD')
loss = pvn.value_model.evaluate(board_test, value_test, batch_size=128)
print('mean squared loss of predicted values:{:.4f}'.format(loss))

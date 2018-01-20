from __future__ import print_function

import numpy as np
from gomokuZero.model.neural_network import PolicyValueNetwork
from gomokuZero.utils import check_load_path

nn_path = 'data/pre_train/test_version_nn_config.json'
pvn = PolicyValueNetwork.load_model(nn_path)

samples = np.load(check_load_path('data/records/yixin_samples.npz'))
board_tensors = samples['board_tensors']
policy_tensors = samples['policy_tensors']

pvn.policy_model.compile(loss='categorical_crossentropy', optimizer='SGD',
                         metrics=['acc'])
loss, acc = pvn.policy_model.evaluate(board_tensors, policy_tensors)
print('accuracy:{:.2f}%'.format(acc*100))

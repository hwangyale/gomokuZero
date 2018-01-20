from __future__ import print_function

import numpy as np
from gomokuZero.utils import check_load_path
from gomokuZero.pre_train.supervised_learning import get_samples_from_history

history_path = 'data/records/yixin_records.npz'
history_pool = np.load(check_load_path(history_path))['history_pool']
samples_path = 'data/records/yixin_samples.npz'
board_tensors, policy_tensors, value_tensors = get_samples_from_history(
    history_pool, augment=True, save_path=samples_path
)
print(board_tensors.shape, policy_tensors.shape, value_tensors.shape)

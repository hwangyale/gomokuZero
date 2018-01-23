import numpy as np


try:
    range = xrange
except NameError:
    pass


def get_rot_func(k):
    def rot_func(tensor):
        dimention = len(tensor.shape)
        if dimention == 2:
            return np.rot90(tensor, k=k)
        elif dimention > 2:
            return np.rot90(tensor, k=k, axes=(dimention-2, dimention-1))
        else:
            raise Exception('dimention of the tensor must be greater than 1')
    return rot_func

rot090 = get_rot_func(1)
rot180 = get_rot_func(2)
rot270 = get_rot_func(3)
rot360 = get_rot_func(4)

def flip_row(tensor):
    return tensor[..., ::-1, :]

def flip_col(tensor):
    return tensor[..., :, ::-1]

def flip_lef(tensor):
    dimention_idxs = list(range(len(tensor.shape)))
    dimention_idxs[-2], dimention_idxs[-1] = dimention_idxs[-1], dimention_idxs[-2]
    return np.transpose(tensor, axes=dimention_idxs)

def flip_rig(tensor):
    return rot270(flip_lef(rot090(tensor)))


roting_fliping_functions = [
    rot090,
    rot180,
    rot270,
    rot360,
    flip_row,
    flip_col,
    flip_lef,
    flip_rig
]

inverse_mapping = {
    rot090: rot270,
    rot180: rot180,
    rot270: rot090,
    rot360: rot360,
    flip_row: flip_row,
    flip_col: flip_col,
    flip_lef: flip_lef,
    flip_rig: flip_rig
}

def augment_data(board_tensors, policy_tensors, value_tensors):
    augment_board_tensors = []
    augment_policy_tensors = []
    augment_value_tensors = []
    for idx, func in enumerate(roting_fliping_functions):
        sys.stdout.write(' '*79 + '\r')
        sys.stdout.flush()
        sys.stdout.write('function index:{:d}\r'.format(idx))
        sys.stdout.flush()
        augment_board_tensors.append(func(board_tensors))
        augment_policy_tensors.append(func(policy_tensors))
        augment_value_tensors.append(value_tensors)

    sys.stdout.write(' '*79 + '\r')
    sys.stdout.flush()

    board_tensors = np.concatenate(augment_board_tensors, axis=0)
    policy_tensors = np.concatenate(augment_policy_tensors, axis=0)
    value_tensors = np.concatenate(augment_value_tensors, axis=0)

    return board_tensors, policy_tensors, value_tensors

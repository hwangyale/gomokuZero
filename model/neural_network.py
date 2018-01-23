import os
import numpy as np
import warnings
from keras import backend as K
from keras import engine as keras_engine
from keras import layers as keras_layers
from keras import regularizers
from ..constant import *
from ..utils import NeuralNetworkBase, NeuralNetworkDecorate, tolist, sample
from ..utils import roting_fliping_functions
from .preprocess import Preprocessor

rng = np.random

@NeuralNetworkDecorate
class PolicyValueNetwork(NeuralNetworkBase):
    def __init__(self, **kwargs):
        self.model, self.policy_model, self.value_model = self.create(**kwargs)
        self.network_setting = kwargs
        self.preprocessor = Preprocessor()

    def get_inputs(self, boards, rot_flip=False):
        boards = tolist(boards)
        if rot_flip:
            funcs = [rng.choice(roting_fliping_functions) for _ in boards]
        else:
            funcs = None
        return self.preprocessor.get_inputs(boards, funcs)

    def get_distribution_values(self, boards, rot_flip=False, **kwargs):
        inputs = self.get_inputs(boards, rot_flip)
        distributions, values = self.forward(inputs, **kwargs)
        return self.preprocessor.get_outputs(distributions, boards), values[:, 0].tolist()

    def get_distributions(self, boards, rot_flip=False, **kwargs):
        inputs = self.get_inputs(boards, rot_flip)
        distributions = self.forward(inputs, self.policy_model, **kwargs)
        return self.preprocessor.get_outputs(distributions, boards)

    def get_values(self, boards, rot_flip=False, **kwargs):
        inputs = self.get_inputs(boards, rot_flip)
        return self.forward(inputs, self.value_model, **kwargs)[:, 0].tolist()

    def get_policy_values(self, boards, rot_flip=False, **kwargs):
        distributions, values = self.get_distribution_values(boards, rot_flip, **kwargs)
        return self.preprocessor.get_policies(distributions, boards), values

    def get_policies(self, boards, rot_flip=False, **kwargs):
        return self.preprocessor.get_policies(self.get_distributions(boards, rot_flip, **kwargs), boards)

    def get_position_values(self, boards, rot_flip=False, **kwargs):
        policies, values = self.get_policy_values(boards, rot_flip, **kwargs)
        return self.generate_positions(policies), values

    def get_positions(self, boards, rot_flip=False, **kwargs):
        policies = self.get_policies(boards, rot_flip, **kwargs)
        return self.generate_positions(policies)

    def generate_positions(self, policies):
        policies = tolist(policies)
        positions = []
        for idx, policy in enumerate(policies):
            positions.append(sample(policy))

        if len(policies) == 1:
            return positions[0]
        else:
            return positions

    @classmethod
    def create(cls, **kwargs):
        default = {
            'blocks': 3,
            'kernel_size': (3, 3),
            'filters': 256,
            'input_shape': Preprocessor.shape[1:],
            'output_size': SIZE**2,
            'weight_decay': 1e-4
        }

        unknown = set(kwargs.keys()) - set(default.keys())
        if unknown:
            raise Exception(('Unknown arguments:'+','.join(['{}']*len(unknown)))
                            .format(*unknown))

        default.update(kwargs)

        conv_setting = {
            'filters': default['filters'],
            'kernel_size': default['kernel_size'],
            'data_format': 'channels_first',
            'padding': 'same',
            'activation': 'linear',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': regularizers.l2(default['weight_decay'])
        }

        input = keras_engine.Input(default['input_shape'])
        tensor = keras_layers.Conv2D(
            name='pre_convolution', **conv_setting
        )(input)
        tensor = keras_layers.BatchNormalization(
            axis=1, name='pre_batch_normalization'
        )(tensor)
        tensor = keras_layers.Activation('relu', name='pre_relu')(tensor)

        def get_block_output(x, count=[0]):
            count[0] += 1
            t = keras_layers.Conv2D(
                name='convolution_{:d}_1'.format(count[0]), **conv_setting
            )(x)
            t = keras_layers.BatchNormalization(
                axis=1, name='batch_normalization_{:d}_1'.format(count[0])
            )(t)
            t = keras_layers.Activation(
                'relu', name='relu_{:d}_1'.format(count[0])
            )(t)

            t = keras_layers.Conv2D(
                name='convolution_{:d}_2'.format(count[0]), **conv_setting
            )(t)
            t = keras_layers.BatchNormalization(
                axis=1, name='batch_normalization_{:d}_2'.format(count[0])
            )(t)

            t = keras_layers.add([x, t], name='add_{:d}'.format(count[0]))
            y = keras_layers.Activation(
                'relu', name='relu_{:d}_2'.format(count[0])
            )(t)

            return y

        for _ in range(default['blocks']):
            tensor = get_block_output(tensor)

        conv_setting.update({'filters': 2, 'kernel_size': (1, 1)})
        policy_tensor = keras_layers.Conv2D(
            name='policy_convolution', **conv_setting
        )(tensor)
        policy_tensor = keras_layers.BatchNormalization(
            axis=1, name='policy_batch_normalization'
        )(policy_tensor)
        policy_tensor = keras_layers.Activation(
            'relu', name='policy_relu'
        )(policy_tensor)
        policy_tensor = keras_layers.Flatten(name='policy_flatten')(policy_tensor)
        policy_output = keras_layers.Dense(
            default['output_size'], activation='softmax', name='p',
            kernel_regularizer=regularizers.l2(default['weight_decay'])
        )(policy_tensor)

        conv_setting['filters'] = 1
        value_tensor = keras_layers.Conv2D(
            name='value_convolution', **conv_setting
        )(tensor)
        value_tensor = keras_layers.BatchNormalization(
            axis=1, name='value_batch_normalization'
        )(value_tensor)
        value_tensor = keras_layers.Activation(
            'relu', name='value_relu'
        )(value_tensor)
        value_tensor = keras_layers.Flatten(name='value_flatten')(value_tensor)
        value_tensor = keras_layers.Dense(
            256, activation='relu', name='value_fc',
            kernel_regularizer=regularizers.l2(default['weight_decay'])
        )(value_tensor)
        value_output = keras_layers.Dense(
            1, activation='tanh', name='v',
            kernel_regularizer=regularizers.l2(default['weight_decay'])
        )(value_tensor)

        model = keras_engine.Model(input, [policy_output, value_output],
                                   name='policy_value_model')

        policy_model = keras_engine.Model(input, policy_output,
                                          name='policy_model')

        value_model = keras_engine.Model(input, value_output, name='value_model')

        return model, policy_model, value_model

    def get_config(self):
        default = self.network_setting
        return default

    def copy(self):
        cache_json_path = '{:s}.json'.format(str(id(self)))
        cache_weights_path = '{:s}.npz'.format(str(id(self)))
        self.save_model(cache_json_path, cache_weights_path)
        model = self.__class__.load_model(cache_json_path)
        os.remove(cache_json_path)
        os.remove(cache_weights_path)
        return model

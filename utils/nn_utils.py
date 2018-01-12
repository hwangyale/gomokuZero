from __future__ import print_function

from abc import ABCMeta, abstractmethod
import json
from ..utils.io_utils import *

NEURAL_NETWORK_CLASSES = {}
def NeuralNetworkDecorate(cls):
    global NEURAL_NETWORK_CLASSES
    NEURAL_NETWORK_CLASSES[cls.__name__] = cls
    return cls


class NeuralNetworkBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.model = self.create(**kwargs)
        #To Do:get forward function

    @abstractmethod
    def create(cls, **kwargs):
        pass

    @staticmethod
    def load_model(path):
        with open(check_load_path(path), 'r') as f:
            object_specs = json.load(f)
        cls_name = object_specs['class']
        nn_cls = NEURAL_NETWORK_CLASSES.get(nn_cls, None)
        if nn_cls is None:
            raise Exception('no class named `{:s}`, '
                            'please check the '
                            'neural network registration'.format(cls_name))
        config = object_specs['config']
        model = nn_cls(**config)
        weights_path = object_specs.get('weights_path')
        if weights_path:
            print('loading weights...')
            self.load_weights(weights_path)
        return model

    def save_model(self, path, weights_path=None):
        path = check_save_path(path)
        object_specs = {
            'class': self.__class__.__name__,
            'config': self.get_config()
        }
        if weights_path:
            object_specs['weights_path'] = self.model.save_weights(weights_path)
        with open(path, 'w') as f:
            json.dump(object_specs, f)

    def load_weights(self, weights_path):
        self.model.load_weights(check_load_path(weights_path), True)

    def save_weights(self, weights_path):
        self.model.save_weights(check_save_path(weights_path))
        return weights_path

    @abstractmethod
    def get_config(self):
        pass

from __future__ import print_function

from abc import ABCMeta, abstractmethod
import json
from ..utils.io_utils import *

NEURAL_NETWORK_CLASSES = {}


def NeuralNetworkDecorate(cls):
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

    @classmethod
    def load_model(cls, path):
        with open(check_load_path(path), 'r') as f:
            object_specs = json.load(f)
        config = object_specs['config']
        model = cls(**config)
        weights_path = object_specs.get('weights_path')
        if weights_path:
            print('load weights')
            model.model.load_weights(check_load_path(weights_path), True)
        return model

    def save_model(self, path, weights_path=None):
        path = check_save_path(path)
        object_specs = {'config': self.get_config()}
        if weights_path:
            self.model.save_weights(check_save_path(weights_path))
            object_specs['weights_path'] = weights_path
        with open(path, 'w') as f:
            json.dump(object_specs, f)

    @abstractmethod
    def get_config(self):
        pass

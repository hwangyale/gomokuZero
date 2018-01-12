from keras import backend as K
from keras import engine as keras_engine
from keras import layers as keras_layers
from ..utils import NeuralNetworkBase, NeuralNetworkDecorate

@NeuralNetworkDecorate
class ActionValueNetwork(NeuralNetworkBase):
    @classmethod
    def create(cls, **kwargs):
        kwargs = {
            'blocks': 3,
            'kernel_size': (5, 5),
            'filters': 64
        }

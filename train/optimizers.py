import json
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
from ..utils import check_save_path

class StochasticGradientDescent(optimizers.SGD):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.moments = moments

        self.weights = [self.iterations] + moments

        #initialize
        if hasattr(self, 'init_weights'):
            for w, init_w in zip(self.weights, self.init_weights):
                K.set_value(w, init_w)

        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = super(StochasticGradientDescent, self).get_config()
        if hasattr(self, 'weights'):
            config['init_weights'] = [K.get_value(w).tolist() for w in self.weights]
        return config

    @classmethod
    def from_config(cls, config):
        init_weights = config.pop('init_weights', None)
        optimizer = cls(**config)
        if init_weights:
            optimizer.init_weights = [np.array(w) for w in init_weights]
        return optimizer


class Adam(optimizers.Adam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        #initialize
        if hasattr(self, 'init_weights'):
            for w, init_w in zip(self.weights, self.init_weights):
                K.set_value(w, init_w)

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = super(Adam, self).get_config()
        if hasattr(self, 'weights'):
            config['init_weights'] = [K.get_value(w).tolist() for w in self.weights]
        return config

    @classmethod
    def from_config(cls, config):
        init_weights = config.pop('init_weights', None)
        optimizer = cls(**config)
        if init_weights:
            optimizer.init_weights = [np.array(w) for w in init_weights]
        return optimizer


class OptimizerSaving(Callback):
    def __init__(self, optimizer, file_path):
        self.optimizer = optimizer
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        with open(check_save_path(self.file_path), 'w') as f:
            config = self.optimizer.get_config()
            json.dump(config, f)

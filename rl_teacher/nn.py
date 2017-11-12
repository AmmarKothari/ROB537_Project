import numpy as np
import tensorflow as tf
import pdb

from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential

class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=int(input_dim)))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        try:
            x = tf.concat([flat_obs, act], axis=1)
        except:
            flat_obs = tf.reshape(flat_obs, shape = (1,flat_obs.shape.dims[1].value))
            act = tf.reshape(act, shape = (1,1))
            x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)

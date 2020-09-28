from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


class DLModel:
    def __init__(self, env, action_size, model_type=1, output_dir=''):
        self.learning_rate = 0.001251
        self.output_dir = output_dir
        self.env = env
        self.action_size = action_size
        if model_type == 1:
            self.model = self._build_model_001()
            self.target_model = self._build_model_001()
        elif model_type == 2:
            self.model = self._build_model_002()
            self.target_model = self._build_model_002()
        elif model_type == 3:
            self.model = self._build_model_003()
            self.target_model = self._build_model_003()
        elif model_type == 4:
            self.model = self._build_model_004()
            self.target_model = self._build_model_004()
        elif model_type == 5:
            self.model = self._build_model_005()
            self.target_model = self._build_model_005()
        elif model_type == 6:
            self.model = self._build_model_006()
            self.target_model = self._build_model_006()
        self.update_target_model()

    def _build_model_001(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(48, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_002(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(150, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(120, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_003(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(300, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(240, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_004(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(48, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        loss = tf.keras.losses.Huber()
        model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'mae'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_005(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(150, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(120, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        loss = tf.keras.losses.Huber()
        model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'mae'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_006(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(300, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(240, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        loss = tf.keras.losses.Huber()
        model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'mae'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    @staticmethod
    def loss_fn(preds, r):
        return -1 * np.sum(r * np.log(preds))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


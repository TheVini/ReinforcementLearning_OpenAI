from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class DLModel:
    def __init__(self,
                 env,
                 action_size,
                 state_size=0,
                 states=0,
                 upper_bound=0,
                 model_type=1,
                 algorithm='dql',
                 output_dir=''):

        self.learning_rate = 0.001251
        self.output_dir = output_dir
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.algorithm = algorithm
        self.states = states

        if self.algorithm == 'dql':
            if model_type == 1:
                self.model = self._build_model_dql_001()
                self.target_model = self._build_model_dql_001()
            elif model_type == 2:
                self.model = self._build_model_dql_002()
                self.target_model = self._build_model_dql_002()
            elif model_type == 3:
                self.model = self._build_model_dql_003()
                self.target_model = self._build_model_dql_003()
            elif model_type == 4:
                self.model = self._build_model_dql_004()
                self.target_model = self._build_model_dql_004()
            elif model_type == 5:
                self.model = self._build_model_dql_005()
                self.target_model = self._build_model_dql_005()
            elif model_type == 6:
                self.model = self._build_model_dql_006()
                self.target_model = self._build_model_dql_006()
            elif model_type == 7:
                self.model = self._build_model_dql_007_mario()
                self.target_model = self._build_model_dql_007_mario()
            self.update_target_model()
        elif self.algorithm == 'ddpg':
            if model_type == 1:
                self.actor_model = self.get_actor_001(state_size, upper_bound)
                self.critic_model = self.get_critic_001(state_size, action_size)

                self.target_actor = self.get_actor_001(state_size, upper_bound)
                self.target_critic = self.get_critic_001(state_size, action_size)

            elif model_type == 2:
                self.actor_model = self.get_actor_002(state_size, upper_bound)
                self.critic_model = self.get_critic_002(state_size, action_size)

                self.target_actor = self.get_actor_002(state_size, upper_bound)
                self.target_critic = self.get_critic_002(state_size, action_size)

            # Making the weights equal initially
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())
        elif self.algorithm == 'sac':
            if model_type == 1:
                print("OK")

    def _build_model_dql_001(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(48, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_dql_002(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(150, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(120, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_dql_003(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(300, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(240, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_dql_004(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(48, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        loss = tf.keras.losses.Huber()
        model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'mae'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_dql_005(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(150, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(120, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        loss = tf.keras.losses.Huber()
        model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'mae'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_dql_006(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(300, input_shape=state_shape, activation=LeakyReLU()))
        model.add(Dense(240, activation=LeakyReLU()))
        model.add(Dense(self.action_size, activation='linear'))
        loss = tf.keras.losses.Huber()
        model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'mae'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def _build_model_dql_007_mario(self):
        model = tf.keras.models.Sequential()
        state_shape = (self.state_size, self.state_size, self.states)

        model.add(Conv2D(32, kernel_size=(8, 8), activation=LeakyReLU(), input_shape=state_shape))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(4, 4), activation=LeakyReLU()))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU()))

        model.add(Flatten())
        model.add(Dense(512, activation=LeakyReLU(), name='layer_1'))
        model.add(Dense(128, activation=ReLU(), name='layer_2'))
        model.add(Dense(self.action_size, activation='linear', name='layer_3'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        plot_model(model, to_file=self.output_dir + '/model.png', show_shapes=True)
        return model

    def get_actor_001(self, num_states, upper_bound):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.action_size, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic_001(self, num_states, num_actions):
        # State as input
        state_input = layers.Input(shape=num_states)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=num_actions)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def get_actor_002(self, num_states, upper_bound):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(600, activation="relu")(inputs)
        out = layers.Dense(300, activation="relu")(out)
        outputs = layers.Dense(self.action_size, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic_002(self, num_states, num_actions):
        # State as input
        state_input = layers.Input(shape=num_states)
        state_out = layers.Dense(600, activation="relu")(state_input)
        state_out = layers.Dense(300, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=num_actions)
        action_out = layers.Dense(300, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(600, activation="relu")(concat)
        out = layers.Dense(300, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def get_value_network_001(self, num_states, hidden_layer_size):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(hidden_layer_size, activation="relu")(inputs)
        out = layers.Dense(hidden_layer_size, activation="relu")(out)
        outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

        model = tf.keras.Model(inputs, outputs)
        return model

    def get_SoftQNetwork_001(self, num_states, num_actions, hidden_layer_size):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        input_size = layers.Concatenate()([num_states, num_actions])
        inputs = layers.Input(shape=(input_size,))
        out = layers.Dense(hidden_layer_size, activation="relu")(inputs)
        out = layers.Dense(hidden_layer_size, activation="relu")(out)
        outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

        model = tf.keras.Model(inputs, outputs)
        return model

    @staticmethod
    def loss_fn(preds, r):
        return -1 * np.sum(r * np.log(preds))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        if self.algorithm == 'dql':
            self.model.save_weights(name)
        elif self.algorithm == 'ddpg':
            self.actor_model.save_weights(name)

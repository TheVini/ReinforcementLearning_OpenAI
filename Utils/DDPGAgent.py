from collections import deque
from Utils import NNModel, Utils
import tensorflow as tf
from skimage import img_as_ubyte
from skimage.transform import resize
from enum import Enum
import numpy as np
import gym
import random
import imageio

"""
Credits to: https://keras.io/examples/rl/ddpg_pendulum/
"""


class OUActionNoise:
    """
    To implement better exploration by the Actor network, we use noisy perturbations,
    specifically an Ornstein-Uhlenbeck process for generating noise, as described in the paper.
    It samples noise from a correlated normal distribution.
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.x_prev = 0

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, num_states, num_actions, gamma, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.gamma = gamma

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch, target_actor, target_critic,
            actor_model, critic_model, actor_optimizer, critic_optimizer
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

        return [actor_loss, critic_loss]

    # We compute the loss and update parameters
    def learn(self, target_actor, target_critic, actor_model, critic_model, actor_optimizer, critic_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return self.update(state_batch, action_batch, reward_batch, next_state_batch,
                           target_actor, target_critic, actor_model, critic_model, actor_optimizer, critic_optimizer)


class DDPGAgent:
    def __init__(self, batch_size=32, success_margin=150, success_score=200,
                 action_size=None, memory_size=None, record_video=False,
                 gym_env='Pendulum-v0', project='Pendulum'):
        Utils.disable_view_window()

        self.record_video = record_video
        self.model_output_dir, self.video_output_dir, self.others_dir = Utils.create_dirs()
        self.success_margin = success_margin
        self.success_score = success_score
        self.project = project

        self.env = gym.make(gym_env)
        self.state_size = self.env.observation_space.shape[0]
        self.batch_size = batch_size

        if action_size is None:
            try:
                self.action_size = self.env.action_space.shape[0]
            except:
                self.action_size = self.env.action_space.n
        else:
            self.action_size = action_size

        # Used to update target networks
        self.tau = 0.005
        self.gamma = 0.99
        self.highest_score = 0

        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        self.rootModel = NNModel.DLModel(self.env, self.action_size, self.state_size,
                                         self.upper_bound, algorithm='ddpg')

        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.buffer = Buffer(self.state_size, self.action_size, self.gamma, memory_size, self.batch_size)

        self.ep_score, self.renders, self.actor_losses, self.critic_losses = [], [], [], []

        general_info = 'Agent info:\n\tBatch size: {}\n\tMemory length: {}\n\tGamma: {}\n\t' \
                       'Success Margin: -{}\n' \
            .format(self.batch_size, memory_size, self.gamma, self.success_margin)
        Utils.log_info(self.others_dir, general_info)

    def remember(self, prev_state, action, reward, state):
        if self.record_video:
            # Append new frame to video memory
            self.renders.append(img_as_ubyte(resize(self.env.render(mode='rgb_array'), (640, 960, 3))))
        # Append set of parameters to memory
        self.buffer.record((prev_state, action, reward, state))
        # Send next state as old state
        return state

    def save_model(self, episode, score):
        """
        Just save a model if its score is higher than the highest score
        """
        if episode == 0:
            self.highest_score = score
        elif score > self.highest_score:
            self.highest_score = score
            self.rootModel.save(self.model_output_dir + '/weights_final' + '{:04d}'.format(episode + 1) + ".hdf5")

    def policy(self, state):
        sampled_actions = tf.squeeze(self.rootModel.actor_model(state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    def test(self, model_path=None):
        if model_path is not None:
            self.rootModel.actor_model.load(model_path)
        test_score_list = []
        test_rounds = 100
        for i in range(test_rounds):
            score = 0.
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while True:
                act_values = self.rootModel.actor_model.predict(state)[0]
                print(act_values)
                next_state, reward, done, _ = self.env.step(act_values[0])
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                state = next_state
                if done:
                    break
            test_score_list.append(score)
            score_text = "Progress: {}/{} | Score: {:.04f} | Avg. Score: {:.04f}".format(
                (i + 1), test_rounds, score, np.mean(test_score_list))
            Utils.test_log_info(self.others_dir, score_text)

    def learn(self):
        actor_loss, critic_loss = self.buffer.learn(self.rootModel.target_actor, self.rootModel.target_critic,
                                                    self.rootModel.actor_model, self.rootModel.critic_model,
                                                    self.actor_optimizer, self.critic_optimizer)
        self.rootModel.update_target(self.rootModel.target_actor.variables, self.rootModel.actor_model.variables,
                                     self.tau)
        self.rootModel.update_target(self.rootModel.target_critic.variables, self.rootModel.critic_model.variables,
                                     self.tau)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

    def is_solved_up(self):
        if len(self.ep_score) > self.success_margin:
            return np.mean(self.ep_score[-self.success_margin]) >= self.success_score
        return False

    def checkout_steps(self, episode, score):
        """
        After all episode ending, a set of tasks is done
        """
        self.ep_score.append(score)
        if self.record_video:
            # Set video name
            video_name = self.video_output_dir + '/{}_{:08d}.mp4'.format(self.project, episode + 1)
            # Record video and clean render list
            imageio.mimwrite(video_name, self.renders, fps=60)
            self.renders.clear()
        text = "Episode {}".format(episode + 1)

        # Plot graphs
        Utils.full_plot(self.ep_score, self.actor_losses, text, self.others_dir, batch_size=self.batch_size)

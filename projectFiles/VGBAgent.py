from collections import deque
from projectFiles import VGBModel, VGBUtils
from skimage import img_as_ubyte
from skimage.transform import resize
from math import pow
from enum import Enum
import numpy as np
import gym
import random
import imageio


class ActionTypeEnum(Enum):
    SimpleAction = 1
    ComplexAction = 2


class DQNAgent:
    def __init__(self, deque_size=None, replay=1, action_type=1, record_video=False):
        VGBUtils.disable_view_window()
        VGBUtils.del_dirs()
        self.actions_dict = {ActionTypeEnum.SimpleAction: 'Simple Action',
                             ActionTypeEnum.ComplexAction: 'Complex Action'}

        self.env = gym.make('LunarLander-v2')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.record_video = record_video
        self.action_type = ActionTypeEnum(action_type)
        self.model_output_dir, self.video_output_dir, self.others_dir = VGBUtils.create_dirs()
        self.replay = replay

        self.highest_score = 0
        self.action = 0
        self.batch_size = 32
        self.epoch = 5
        self.memory = deque(maxlen=self.batch_size) if deque_size is None else deque(maxlen=deque_size)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = .95
        self.epsilon_min = 0.05
        self.DLModel = VGBModel.DLModel(self.env, self.action_size, self.others_dir)
        self.ep_score, self.renders, self.losses = [], [], []

        general_info = 'Agent info:\n\tBatch size: {}\n\tEpoch(s): {}\n\t' \
                       'Memory length: {}\n\tGamma: {}\n\tEpsilon decay: {}\n\tReplay: {:03d}\n\t' \
                       'Action Type: {}\n'\
            .format(self.batch_size, self.epoch, self.memory.maxlen, self.gamma,
                    self.epsilon_decay, self.replay, self.actions_dict[self.action_type])
        VGBUtils.log_info(general_info)

    def remember(self, state, action, reward, next_state, done):
        if self.record_video:
            # Append new frame to video memory
            self.renders.append(img_as_ubyte(resize(self.env.render(mode='rgb_array'), (640, 960, 3))))
        # Append set of parameters to memory
        self.memory.append((state, action, reward, next_state, done))
        # Send next state as old state
        return next_state

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def checkout_steps(self, episode, score):
        """
        After all episode ending, a set of tasks is done
        """
        self.DLModel.update_target_model()
        self.ep_score.append(score)
        if self.record_video:
            # Set video name
            video_name = self.video_output_dir + '/Lander_{:08d}.mp4'.format(episode + 1)
            # Record video and clean render list
            imageio.mimwrite(video_name, self.renders, fps=60)
            self.renders.clear()
        text = "Episode {}".format(episode + 1)
        # Plot graphs
        VGBUtils.full_plot(self.ep_score, self.losses, text, self.others_dir, batch_size=self.batch_size)
        self.update_epsilon()
        self.clear_memory()

    def save_model(self, episode, score):
        """
        Just save a model if its score is higher than the highest score
        """
        if episode == 0:
            self.highest_score = score
        elif score > self.highest_score:
            self.highest_score = score
            self.DLModel.save(self.model_output_dir + '/weights_final' + '{:04d}'.format(episode + 1) + ".hdf5")

    def clear_memory(self):
        if self.replay == 2:
            self.memory.clear()

    def _take_action(self):
        next_state, reward, done, _ = self.env.step(self.action)
        return [next_state, reward, done, _, self.action]

    def _choose_simple_action(self, state):
        if np.random.rand(1) <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.DLModel.model.predict(state)
        self.action = np.argmax(act_values[0])

    def _choose_complex_ep_greedy_act(self, state):
        """
        Save an action based on weighted average of available actions
        :param state: current state
        """
        act_values = self.DLModel.model.predict(state)
        prob = VGBUtils.softmax(VGBUtils.normalize(act_values)[0])
        self.action = np.random.choice(self.action_size, 1, replace=False, p=prob)[0]

    def _choose_take_simple_action(self, state):
        self._choose_simple_action(state)
        return self._take_action()

    def _choose_take_complex_action(self, state):
        if self.epsilon > self.epsilon_min:
            self._choose_simple_action(state)
        else:
            self._choose_complex_ep_greedy_act(state)
        return self._take_action()

    def choose_and_take_action(self, state):
        if self.action_type == ActionTypeEnum.SimpleAction:
            return self._choose_take_simple_action(state)
        if self.action_type == ActionTypeEnum.ComplexAction:
            return self._choose_take_complex_action(state)

    def train(self):
        if len(self.memory) >= self.batch_size:
            loss = 0
            if self.replay == 1:
                loss = self.replay_001(self.batch_size)
            elif self.replay == 2:
                loss = self.replay_002(self.batch_size)
            self.losses.append(loss)

    def discount_rewards(self, rewards):
        disc_return = []
        [disc_return.append(reward * pow(self.gamma, index)) for index, reward in enumerate(rewards)]
        disc_return /= max(disc_return)
        return disc_return

    def replay_001(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                aux = np.amax(self.DLModel.target_model.predict(next_state)[0])
                target = (reward + self.gamma * aux)
            target_f = self.DLModel.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])

        history = self.DLModel.model.fit(np.array(states), np.array(targets), epochs=self.epoch, verbose=0)
        return np.mean(history.history['loss'])

    def replay_002(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        targets = []
        states = []

        reward_batch = [reward for state, action, reward, next_state, done in minibatch]
        reward_batch = self.discount_rewards(reward_batch)

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            target_f = self.DLModel.model.predict(state)
            target_f[0][action] *= reward_batch[index]
            targets.append(target_f[0])
            states.append(state[0])

        history = self.DLModel.model.fit(np.array(states), np.array(targets), epochs=self.epoch, verbose=0)
        return np.mean(history.history['loss'])

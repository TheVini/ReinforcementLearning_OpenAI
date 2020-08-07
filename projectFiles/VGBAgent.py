from collections import deque
from projectFiles import VGBModel, VGBUtils
from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
import gym
import random
import imageio


class DQNAgent:
    def __init__(self, delete_dirs=False):
        VGBUtils.disable_view_window()
        self.env = gym.make('LunarLander-v2')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        if delete_dirs:
            VGBUtils.del_dirs()
        self.model_output_dir, self.video_output_dir, self.others_dir = VGBUtils.create_dirs()
        self.highest_score = 0
        self.batch_size = 32
        self.memory = deque(maxlen=200000)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = .95
        self.epsilon_min = 0.00001
        self.DLModel = VGBModel.DLModel(self.env, self.action_size, self.others_dir)
        self.ep_score, self.renders, self.losses = [], [], []

    def remember(self, state, action, reward, next_state, done):
        # Append new frame to video memory
        self.renders.append(img_as_ubyte(resize(self.env.render(mode='rgb_array'), (640, 960, 3))))
        # Append set of parameters to memory
        self.memory.append((state, action, reward, next_state, done))
        # Send next state as old state
        return next_state

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def checkout_steps(self, epoch, score):
        """
        After all episode ending, a set of tasks is done
        """
        self.DLModel.update_target_model()
        self.ep_score.append(score)
        # Set video name
        video_name = self.video_output_dir + '/Lander_{:08d}.mp4'.format(epoch + 1)
        # Record video and clean render list
        imageio.mimwrite(video_name, self.renders, fps=60)
        self.renders.clear()
        text = "Época {}".format(epoch + 1)
        # Plot graphs
        VGBUtils.full_plot(self.ep_score, self.losses, text, self.others_dir)
        self.update_epsilon()

    def save_model(self, epoch, score):
        """
        Just save a model if its score is higher than the highest score
        """
        if epoch == 0:
            self.highest_score = score
        elif score > self.highest_score:
            self.highest_score = score
            self.DLModel.save(self.model_output_dir + '/weights_final' + '{:04d}'.format(epoch + 1) + ".hdf5")

    def act(self, state):
        if np.random.rand(1) <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.DLModel.model.predict(state)
        return np.argmax(act_values[0])

    def train_001(self):
        if len(self.memory) > self.batch_size:
            loss = self.replay_001(self.batch_size)
            self.losses.append(loss)

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

        history = self.DLModel.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        return np.mean(history.history['loss'])

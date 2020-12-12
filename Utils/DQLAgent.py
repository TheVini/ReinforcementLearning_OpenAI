from collections import deque
from Utils import NNModel, Utils
from skimage import img_as_ubyte
from skimage.transform import resize
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import gym
import random
import imageio


class ActionTypeEnum(Enum):
    SimpleAction = 1
    ComplexAction = 2


class DQLAgent:
    def __init__(self, action_type=1, batch_size=32,
                 model_type=1, success_margin=150, success_score=200,
                 action_size=None, memory_size=None, record_video=False, record_video_on_test=False, target_model=False,
                 gym_env=None, project=None):
        Utils.disable_view_window()
        self.actions_dict = {ActionTypeEnum.SimpleAction: 'Simple Action',
                             ActionTypeEnum.ComplexAction: 'Complex Action'}

        if gym_env is not None:
            self.env = gym.make(gym_env)
            self.state_size = self.env.observation_space.shape[0]

            if action_size is None:
                try:
                    self.action_size = self.env.action_space.shape[0]
                except:
                    self.action_size = self.env.action_space.n
            else:
                self.action_size = action_size
            self.action = self.env.action_space.sample()

        self.record_video = record_video
        self.record_video_on_test = record_video_on_test
        self.action_type = ActionTypeEnum(action_type)
        self.model_output_dir, self.video_output_dir, self.others_dir = Utils.create_dirs()
        self.target_model = target_model

        self.highest_score = 0
        self.batch_size = batch_size
        self.epoch = 1

        self.memory = deque(maxlen=None) if memory_size is None else deque(maxlen=memory_size)
        self.gamma = 0.99
        self.success_margin = success_margin
        self.success_score = success_score
        self.project = project

        self.epsilon = 1.0
        self.epsilon_decay = .99
        self.epsilon_min = 0.01
        if gym_env is not None:
            self.DLModel = NNModel.DLModel(self.env, self.action_size, model_type, self.others_dir, algorithm='dql')
        self.ep_score, self.renders, self.test_renders, self.losses = [], [], [], []

        general_info = 'Agent info:\n\tBatch size: {}\n\tEpoch(s): {}\n\t' \
                       'Memory length: {}\n\tGamma: {}\n\tEpsilon decay: {}\n\t' \
                       'Success Margin: -{}\n\tAction Type: {}\n\tTarget model: {}\n' \
            .format(self.batch_size, self.epoch, self.memory.maxlen, self.gamma,
                    self.epsilon_decay, self.success_margin, self.actions_dict[self.action_type], self.target_model)
        Utils.log_info(self.others_dir, general_info)

    def append_new_frame(self):
        """ Save generated env's frame """
        self.renders.append(img_as_ubyte(resize(self.env.render(mode='rgb_array'), (640, 960, 3))))

    def append_new_test_frame(self, epoch=0):
        if self.record_video_on_test:
            # Render image
            image = self.env.render(mode='rgb_array')
            # Transform array to image
            image = Image.fromarray(image, 'RGB')
            # Load font
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 16)
            # Create Draw object
            d = ImageDraw.Draw(image)
            # Write on image
            img_text = "Teste: {:03d}".format(epoch)
            d.text((10, 10), img_text, fill=(255, 255, 255), font=font)
            # Image to Array
            image = np.array(image)
            image = resize(image, (640, 960, 3))
            self.test_renders.append(img_as_ubyte(image))

    def remember(self, state, action, reward, next_state, done):
        if self.record_video:
            # Append new frame to video memory
            self.append_new_frame()
        # Append set of parameters to memory
        self.memory.append((state, action, reward, next_state, done))
        # Send next state as old state
        return next_state

    def forget(self):
        self.memory.clear()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def checkout_steps(self, episode, score):
        """
        After all episode ending, a set of tasks is done
        """
        # Update in each episode ending
        self.DLModel.update_target_model()
        self.ep_score.append(score)
        if self.record_video:
            # Set video name
            video_name = self.video_output_dir + '/{}_{:08d}.mp4'.format(self.project, episode + 1)
            # Record video and clean render list
            imageio.mimwrite(video_name, self.renders, fps=60)
            self.renders.clear()
        text = "Episode {}".format(episode + 1)
        # Plot graphs
        Utils.full_plot(self.ep_score, self.losses, text, self.others_dir, batch_size=self.batch_size)
        self.update_epsilon()

    def save_model(self, episode, score):
        """
        Just save a model if its score is higher than the highest score
        """
        if episode == 0:
            self.highest_score = score
        elif score > self.highest_score:
            self.highest_score = score
            self.DLModel.save(self.model_output_dir + '/weights_final' + '{:04d}'.format(episode + 1) + ".hdf5")

    def _take_action(self):
        next_state, reward, done, info = self.env.step(self.action)
        return [next_state, reward, done, info, self.action]

    def _choose_simple_action(self, state):
        if np.random.rand(1) <= self.epsilon:
            self.action = random.randrange(self.action_size)
        else:
            act_values = self.DLModel.model.predict(state)[0]
            self.action = np.argmax(act_values)

    def _choose_complex_ep_greedy_act(self, state):
        """
        Save an action based on weighted average of available actions
        :param state: current state
        """
        act_values = self.DLModel.model.predict(state)
        prob = Utils.softmax(Utils.normalize(act_values)[0])
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

    def is_solved_up(self):
        if len(self.ep_score) > self.success_margin:
            return np.mean(self.ep_score[-self.success_margin]) >= self.success_score
        return False

    def is_solved_down(self):
        if len(self.ep_score) > self.success_margin:
            return np.mean(self.ep_score[-self.success_margin]) <= self.success_score
        return False

    def choose_and_take_action(self, state):
        if self.action_type == ActionTypeEnum.SimpleAction:
            return self._choose_take_simple_action(state)
        elif self.action_type == ActionTypeEnum.ComplexAction:
            return self._choose_take_complex_action(state)

    def test(self, model_path=None):
        if model_path is not None:
            self.DLModel.load(model_path)
        test_score_list = []
        test_rounds = 100
        for i in range(test_rounds):
            score = 0.
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while True:
                self.append_new_test_frame(i+1)
                act_values = self.DLModel.model.predict(state)[0]
                next_state, reward, done, _ = self.env.step(np.argmax(act_values))
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                state = next_state
                if done:
                    break
            if self.record_video_on_test:
                # Set video name
                video_name = self.video_output_dir + '/{}_test_{:08d}.mp4'.format(self.project, i + 1)
                # Record video and clean render list
                imageio.mimwrite(video_name, self.test_renders, fps=60)
                self.test_renders.clear()
            test_score_list.append(score)
            score_text = "Progress: {}/{} | Score: {:.04f} | Avg. Score: {:.04f}".format(
                (i + 1), test_rounds, score, np.mean(test_score_list))
            Utils.test_log_info(self.others_dir, score_text)

    def train_during_episode(self):
        if len(self.memory) >= self.batch_size:
            self.replay_dqn()

    def replay_dqn(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        if self.target_model:
            targets = rewards + self.gamma * (np.amax(self.DLModel.target_model.predict_on_batch(next_states),
                                                      axis=1)) * (1 - dones)
        else:
            targets = rewards + self.gamma * (np.amax(self.DLModel.model.predict_on_batch(next_states),
                                                      axis=1)) * (1 - dones)

        targets_full = self.DLModel.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        history = self.DLModel.model.fit(states, targets_full, epochs=1, verbose=0)
        self.losses.append(np.mean(history.history['loss']))

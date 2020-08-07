import random, imageio, gym, os, math
import numpy as np
import tensorflow as tf
from collections import deque
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
from IPython.display import clear_output
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.env= env
        self.state_size= state_size
        self.action_size= action_size
        
        self.memory= deque(maxlen=200000)
        
        self.gamma= 0.99
        
        self.epsilon = 1.0
        self.epsilon_decay= .85
        #self.epsilon_decay= .9995
        self.epsilon_min=0.00001
        
        self.learning_rate= 0.001251
        self.model= self._build_model()
        self.target_model=self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        model= tf.keras.models.Sequential()
        state_shape= self.env.observation_space.shape
        model.add(tf.keras.layers.Dense(24, input_shape= state_shape, activation='relu'))
        model.add(tf.keras.layers.Dense(48, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        #model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        
        return model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand(1) <=self.epsilon:
            return random.randrange(self.action_size)
        act_values= self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states=[]
        targets=[]
        
        for state, action, reward, next_state, done in minibatch:
            target=reward
            if not done:
                aux = np.amax(self.target_model.predict(next_state)[0])
                target = (reward + self.gamma * aux)
            target_f= self.model.predict(state)
            target_f[0][action]= target
            states.append(state[0])
            targets.append(target_f[0])
            
        #self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, callbacks=self.callbacks_list)
        #self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        self.model.fit(np.array(states), np.array(targets), epochs=5, verbose=0)
            
        
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

def plotar_graficos(ep_score, text):
    if len(ep_score) > 0:
        clear_output(wait=True)
        ep_score = np.array(ep_score)
        #avg_score = running_mean(ep_score, 50)

        fig = plt.figure(figsize=(10,7))
        fig.suptitle(text, fontsize=14, fontweight='bold')
        plt.ylabel("Score",fontsize=22)
        plt.xlabel("Training Epochs",fontsize=22)
        plt.plot(ep_score, color='green')
        plt.show()
        
def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor

disable_view_window()

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size= env.action_space.n
batch_size = 32
n_episodes= 70000
output_dir = 'model_output'
video_dir = 'videos'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
    
agent= DQNAgent(state_size, action_size)

done = False
counter=0 
scores_memory= deque(maxlen=100)
ep_score = []
for e in range(n_episodes):
    score = 0.
    renders = []
    state=env.reset()
    state= np.reshape(state, [1, state_size])
    
    for time in range(7000):
        action = agent.act(state)
        next_state, reward, done, halp =env.step(action)
        score += reward
        next_state = np.reshape(next_state, [1, state_size])
        renders.append(img_as_ubyte(resize(env.render(mode='rgb_array'),(640,960,3))))
        agent.remember(state, action, reward, next_state, done)
            
        if len(agent.memory)>batch_size:
            agent.replay(batch_size)

        
        state = next_state

        if done:
            scores_memory.append(time)
            scores_avg= np.mean(scores_memory)*-1

            
            print('episode: {}/{}, score: {}, e {:.2}, help: {}, reward: {}, 100score avg: {}'.format(e, n_episodes, time, agent.epsilon, state, reward, scores_avg))

            break
    agent.update_target_model()
    ep_score.append(score)
    
    video_name = video_dir + '/Lander_{:08d}.mp4'.format(e+1)
    imageio.mimwrite(video_name, renders, fps=60)
    if e != 0 and e % 10 == 0:
        text = "Época {}".format(e) 
        plotar_graficos(ep_score, text)
        
    if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
    if e % 50==0:
        agent.save(output_dir + '/weights_final' + '{:04d}'.format(e) + ".hdf5")
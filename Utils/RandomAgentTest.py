from gym import envs
import gym


class RandomAgent:
    def __init__(self, gym_env="CartPole-v1"):
        self.env = gym.make(gym_env)
        self.observation = self.env.reset()
        try:
            self.action_size = self.env.action_space.shape[0]
        except:
            self.action_size = self.env.action_space.n
        print("Observation space: {}\nAction Space: {}".format(self.env.observation_space.shape[0],
                                                               self.action_size))

    def run(self):
        for i in range(1000):
            self.env.render()
            action = self.env.action_space.sample()
            if i == 0:
                print("Action example: ".format(action))
            observation, reward, done, info = self.env.step(action)

            if done:
                self.observation = self.env.reset()
        self.env.close()


def list_envs():
    print(envs.registry.all())

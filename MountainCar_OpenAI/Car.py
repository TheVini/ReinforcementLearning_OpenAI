from Utils import DQNAgent, Utils, RandomAgentTest
from datetime import datetime
import numpy as np

'''
agent = RandomAgentTest.RandomAgent(gym_env="Acrobot-v1")
agent.run()
'''

n_episodes = 400
duration = 200
action = 2
actions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
agent = DQNAgent.DQNAgent(action_type=action, batch_size=128, model_type=3, success_margin=100, success_score=-110,
                          record_video=True, target_model=True, memory_size=None, algorithm="dqn",
                          gym_env="MountainCar-v0", project="MountainCar")
start = datetime.now().time().strftime('%H:%M:%S')

for e in range(n_episodes):
    score = 0.
    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])

    for time in range(duration):
        next_state, reward, done, _, action = agent.choose_and_take_action(state)

        # Trick
        reward = 1 if next_state[0] > 0 else reward

        score += reward
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = agent.remember(state[0], action, reward, next_state, done)
        agent.train_during_episode()

        if done or time == (duration - 1) or reward < -100:
            end = datetime.now().time().strftime('%H:%M:%S')
            total_time = (datetime.strptime(end, '%H:%M:%S') - datetime.strptime(start, '%H:%M:%S'))
            output_text = 'Episode: {}/{}, rounds: {}, epsilon: {:.4}, ' \
                          'score: {:.4}, reward: {:}, elapsed time: {}'. \
                format(e + 1, n_episodes, time, agent.epsilon, score, reward, str(total_time))
            Utils.log_info(agent.others_dir, output_text)
            agent.save_model(e, score)
            break
    agent.checkout_steps(e, score)
    if agent.is_solved_up():
        success_text = f"Task was solved at episode {e}"
        Utils.log_info(agent.others_dir, success_text)
        agent.test()
        break

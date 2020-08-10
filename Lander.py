from projectFiles import VGBAgent, VGBUtils
from datetime import datetime
import numpy as np


n_episodes = 300
agent = VGBAgent.DQNAgent(deque_size=32, replay=2, record_video=True)
start = datetime.now().time().strftime('%H:%M:%S')

for e in range(n_episodes):
    score = 0.
    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])

    for time in range(7000):
        next_state, reward, done, _, action = agent.choose_and_take_action(state)
        score += reward
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = agent.remember(state, action, reward, next_state, done)
        agent.train()

        if done:
            end = datetime.now().time().strftime('%H:%M:%S')
            total_time = (datetime.strptime(end, '%H:%M:%S') - datetime.strptime(start, '%H:%M:%S'))
            output_text = 'Episode: {}/{}, episodes: {}, epsilon: {:.4}, ' \
                          'score: {:.4}, reward: {:}, elapsed time: {}'.\
                format(e+1, n_episodes, time, agent.epsilon, score, reward, str(total_time))
            print(output_text)
            VGBUtils.log_info(output_text)
            agent.save_model(e, score)
            break
    agent.checkout_steps(e, score)

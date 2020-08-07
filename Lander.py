from projectFiles import VGBAgent
import numpy as np

n_episodes = 10000
agent = VGBAgent.DQNAgent(delete_dirs=True)

for e in range(n_episodes):
    score = 0.
    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])

    for time in range(7000):
        next_state, reward, done, _, action = agent.choose_take_complex_action(state)
        score += reward
        next_state = np.reshape(next_state, [1, agent.state_size])

        state = agent.remember(state, action, reward, next_state, done)
        agent.train_001()

        if done:
            print('episode: {}/{}, episodes: {}, epsilon {:.4}, score: {:.4}, reward {:}'
                  .format(e+1, n_episodes, time, agent.epsilon, score, reward))
            agent.save_model(e, score)
            break
    agent.checkout_steps(e, score)

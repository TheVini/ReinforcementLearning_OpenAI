from Utils import DQLAgent, Utils, NNModel
from datetime import datetime
import numpy as np


n_episodes = 400
duration = 7000
action = 1

agent = DQLAgent.DQLAgent(action_type=action, batch_size=128, model_type=4, success_margin=250,
                          success_score=200, record_video=False, record_video_on_test=True, target_model=True,
                          gym_env='LunarLander-v2', project='Lander')
start = datetime.now().time().strftime('%H:%M:%S')

# agent.test()
agent.test('Analises/Analise_3/output_files_007/models/weights_final0287.hdf5')

'''
for e in range(n_episodes):
    score = 0.
    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])

    for time in range(duration):
        next_state, reward, done, _, action = agent.choose_and_take_action(state)
        score += reward
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = agent.remember(state, action, reward, next_state, done)
        agent.train_during_episode()

        if done or time == (duration-1) or reward < -100:
            end = datetime.now().time().strftime('%H:%M:%S')
            total_time = (datetime.strptime(end, '%H:%M:%S') - datetime.strptime(start, '%H:%M:%S'))
            output_text = 'Episode: {}/{}, rounds: {}, epsilon: {:.4}, ' \
                          'score: {:.4}, reward: {:}, elapsed time: {}'.\
                format(e+1, n_episodes, time, agent.epsilon, score, reward, str(total_time))
            Utils.log_info(agent.others_dir, output_text)
            agent.save_model(e, score)
            break
    agent.checkout_steps(e, score)
    if agent.is_solved_up():
        success_text = f"Task was solved at episode {e}"
        Utils.log_info(agent.others_dir, success_text)
        agent.test()
        break
'''
from Utils import DQLMarioAgent, Utils, RandomAgentTest
from datetime import datetime
import numpy as np

'''
agent = RandomAgentTest.RandomAgent(gym_env="SuperMarioBros-v0")
agent.run()
'''

n_episodes = 400
duration = 8000
action = 2
max_episode_len = 100
min_progress = 0.05
x_pos_sync_rate = 10
agent = DQLMarioAgent.DQLMarioAgent(action_type=action, batch_size=64, model_type=2, success_margin=150,
                                    success_score=-100, record_video=True, target_model=True, memory_size=None,
                                    project="Mario", wrapper_type='SIMPLE')
start = datetime.now().time().strftime('%H:%M:%S')

for e in range(n_episodes):
    score = 0.
    state = agent.env.reset()

    last_x_pos = agent.get_first_x_pos()
    last_state = agent.get_first_state()
    current_x_pos = agent.get_first_x_pos()

    state = np.reshape(state, [1, agent.state_size, agent.state_size])

    for time in range(duration):
        next_state, reward, done, info, action = agent.choose_and_take_action(state)
        score += reward
        agent.update_max_distance(info['x_pos'])
        current_x_pos = info['x_pos']

        next_state = agent.prepare_state(next_state, 3)
        next_state = np.reshape(next_state, [1, agent.state_size, agent.state_size])
        state = agent.remember(state, action, reward, next_state, done)
        agent.train_during_episode()

        # Controle pra saber se o Mário ficou travando em alguma posição "x" por muito tempo
        if time % max_episode_len == 0 and time != 0 and \
                (current_x_pos - last_x_pos) < (min_progress * max_episode_len):
            done = True

        if time % x_pos_sync_rate == 0:
            last_x_pos = current_x_pos

        if done or time == (duration - 1):
            end = datetime.now().time().strftime('%H:%M:%S')
            total_time = (datetime.strptime(end, '%H:%M:%S') - datetime.strptime(start, '%H:%M:%S'))
            output_text = 'Episode: {}/{}, rounds: {}, epsilon: {:.4}, max_distance: {}' \
                          'score: {:.4}, reward: {:}, elapsed time: {}, max_episode_len: {}, min_progress: {}, ' \
                          'x_pos_sync_rate: {}'. \
                format(e + 1, n_episodes, time, agent.epsilon, agent.max_distance, score, reward,
                       str(total_time), max_episode_len, min_progress, x_pos_sync_rate)
            Utils.log_info(agent.others_dir, output_text)
            agent.save_model(e, score)
            break
    agent.checkout_steps(e, score)
    if agent.is_solved_up():
        success_text = f"Task was solved at episode {e}"
        Utils.log_info(agent.others_dir, success_text)
        agent.test()
        break

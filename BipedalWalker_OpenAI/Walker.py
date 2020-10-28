import tensorflow as tf
from Utils import DDPGAgent, Utils, RandomAgentTest
from datetime import datetime
import time

'''
agent = RandomAgentTest.RandomAgent(gym_env="BipedalWalker-v3")
agent.run()
'''

'''
agent = DDPGAgent.DDPGAgent(batch_size=128, success_score=0, memory_size=50000, model_type=2,
                            record_video=False, gym_env='BipedalWalker-v3')
start = datetime.now().time().strftime('%H:%M:%S')

n_episodes = 200
duration = 7000
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for e in range(n_episodes):

    prev_state = agent.env.reset()
    score = 0

    for time in range(duration):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = agent.policy(tf_prev_state)[0]
        # Recieve state and reward from environment.
        state, reward, done, info = agent.env.step(action)

        prev_state = agent.remember(prev_state, action, reward, state)
        score += reward
        agent.learn()

        # End this episode when `done` is True
        if done:
            end = datetime.now().time().strftime('%H:%M:%S')
            total_time = (datetime.strptime(end, '%H:%M:%S') - datetime.strptime(start, '%H:%M:%S'))
            output_text = 'Episode: {}/{}, rounds: {}, ' \
                          'score: {:.4}, reward: {:}, elapsed time: {}'. \
                format(e + 1, n_episodes, time, score, reward, str(total_time))
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



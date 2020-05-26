import sys
import numpy as np
import math
import random


import matplotlib.pyplot as plt

NUM_BUCKETS = 64

NUM_ACTIONS = 3
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2
DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
print(DECAY_FACTOR)

NUM_EPISODES = 9999999
MAX_T = 2000

q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)



def train():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99
    total_reward = 0
    total_rewards = []
    training_done = False
    threshold = 1000
    env.set_view(True)
    print("episode, step, total_reward, explore_rate, learning_rate")
    for episode in range(NUM_EPISODES):

        total_rewards.append(total_reward)
        if episode == 50000:
            plt.plot(total_rewards)
            plt.ylabel('rewards')
            plt.show()
            env.save_memory('50000')
            break

        obv = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate)
            obv, reward, done, _ = env.step(action)
            state = state_to_bucket(obv)
            env.remember(state_0, action, reward, state, done)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state
            env.render()
            if done or t >= MAX_T - 1:
                print(episode, t, total_reward, explore_rate, learning_rate)
                break
        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)




def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q_table[state]))
    return action



def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))




def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)



if __name__ == "__main__":


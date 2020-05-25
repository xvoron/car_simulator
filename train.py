from environment import Environment
from ai import DQN_agent
import time
WIDTH = 1600
HEIGTH = 800

min_explore_rate = 0.001
min_learn_rate = 0.2


def ai_race():
    dqn_agent = DQN_agent("success.model")
    env = Environment(WIDTH, HEIGTH)
    cur_state = env.reset()

    while True:
        action = dqn_agent.act(cur_state)
        new_state, reward, done, _ = env.step(action)
        env.draw_all()
        if done:
            break

def train():
    gamma = .9
    epsilon = .95

    trials = 1000 #10_000
    trial_len = 5000


    dqn_agent = DQN_agent()
    env = Environment(WIDTH, HEIGTH)
    steps = []



    print("episode, step, total_reward, epsilon")
    for trial in range(trials):

        cur_state = env.reset() # .reshape(1,2) # TODO
        # print("[DEBUG-train.py] cur_state: {}".format(cur_state))
        total_reward = 0
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action) # TODO not nessasery _
            total_reward += reward
            env.draw_all()


            # reward = reward if not done else -20
            # new_state = new_state # .reshape(1,2) # TODO
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()
            # dqn_agent.target_train()

            cur_state = new_state

            if done:
                break

        dqn_agent.target_train()
        # dqn_agent.replay()
        print(trial, step, total_reward, dqn_agent.epsilon)

        # if step >= 199:
        #     print("Failed to complete in trial {}".format(trial))

        #     if step % 10 == 0:
        #         dqn_agent.save_model("models/trial-{}.model".format(trial))
        # else:
        #     print("Completed in {} trials".format(trial))
        #     dqn_agent.save_model("success.model")
        #     break
        if trial % 100 == 0:
            dqn_agent.save_model("success.model".format(trial))

    print("[INFO] Saving model...")
    dqn_agent.save_model("success.model")

if __name__ == "__main__":
    pass


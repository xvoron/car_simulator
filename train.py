from environment import Environment
from ai import DQN_agent
import time
WIDTH = 1600
HEIGTH = 800


def train():
    gamma = .9
    epsilon = .95

    trials = 1000
    trial_len = 5000


    dqn_agent = DQN_agent()
    env = Environment(WIDTH, HEIGTH)
    steps = []


    for trial in range(trials):

        cur_state = env.reset() # .reshape(1,2) # TODO
        # print("[DEBUG-train.py] cur_state: {}".format(cur_state))
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            print("[DEBUG] {} action: {}".format(__name__, action))
            new_state, reward, done = env.step(action) # TODO not nessasery _
            env.draw_all()


            # reward = reward if not done else -20
            new_state = new_state # .reshape(1,2) # TODO
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()

            cur_state = new_state

            print("[DEBUG] step: {}".format(step))
            if done:
                break

        print("[DEBUG] trial: {}, step: {}".format(trial, step))

        # if step >= 199:
        #     print("Failed to complete in trial {}".format(trial))

        #     if step % 10 == 0:
        #         dqn_agent.save_model("models/trial-{}.model".format(trial))
        # else:
        #     print("Completed in {} trials".format(trial))
        #     dqn_agent.save_model("success.model")
        #     break
        if trial % 10 == 0:
            dqn_agent.save_model("models/trial-{}.model".format(trial))

    dqn_agent.save_model("success.model")

if __name__ == "__main__":
    pass


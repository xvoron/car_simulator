from environment import Environment
from ai import DQN_agent

def train():
    env = Environment()
    gamma = .9
    epsilon = .95

    trials = 1000
    trial_len = 500


    dqn_agent = DQN_agent(env=env)
    steps = []

    for trial in range(trials):
        cur_state = env.reset().reshape(1,2) # TODO
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)


            # reward = reward if not done else -20
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()

            cur_state = new_state

            if done:
                break

        if step >= 199:
            print("Failed to complete in trial {}".format(trial))

            if step % 10 == 0:
                dqn_agent.save_model("models/trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    pass


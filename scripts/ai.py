"""AI module.

Implementation DQN_agent.
"""

import numpy as np
import random
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import time

from collections import deque

"""
New DQN_agent from
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
"""

class DQN_agent():
    """DQN_agent implementation.
    """

    def __init__(self, mode_or_load_model="train"):

        self.mode = mode_or_load_model # Train or load model

        if self.mode == "train":
            print("[INFO] train mode")
            self.memory = deque(maxlen=50000)

            self.gamma = 0.9
            self.epsilon = 1.0
            self.epsilon_min = .01
            self.epsilon_decay = .9998
            self.learning_rate = .01
            self.tau = .125

            self.model = self.create_model()
            self.target_model = self.create_model()

        else:
            print("[INFO] ai race mode")
            self.model = load_model(mode_or_load_model)
            print(self.model.summary())
            print("[INFO] Model was loaded")


        time.sleep(5)


    def create_model(self):
        """Create a NN model.

        argument:
            None
        return:
            - model: NN model.
        """

        model = Sequential()
        model.add(Dense(24, input_dim=7, activation="relu")) # 7 sensors-rays
        model.add(Dense(48, activation="relu"))
        model.add(Dense(3, activation="linear"))             # 3 action
        model.compile(loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print("[INFO] model was created:")
        print(model.summary())
        return model

    def action_sample(self):
        """Random action from [0,1,2]

        arguments:
            None
        return:
            - random number from [0,1,2]
        """
        return np.random.randint(3)

    def act(self, state):
        """Epsilon greedy policy.

        argumets:
            - state: state from environment.
        return:
            action: random or from model prediction.
        """
        if self.mode == "train":
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.random() < self.epsilon:
                return self.action_sample()
            return np.argmax(self.model.predict(state)[0])
        else:
            return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        """Add outputs from environment to memory.

        arguments:
            - state: current state s_n.
            - action: from agent.
            - reward: by action on current state s_n.
            - new_state: s_{n+1}.
            - done: if step done or not ???? # TODO.
        return:
            None
            """
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """Replay states and fit model.

        arguments:
            None
        return:
            None
            """

        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0, shuffle=False)

    def target_train(self):
        """Copy weights from model to target_model
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 -
                    self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        """Save current target_model.
        """
        self.target_model.save(fn)



"""
AI module.

@autor: Artyom Voronin

"""
import numpy as np
import random
import os



class Network():
    def __init__(self):
        pass

class ReplayMemory():

    def __init__(self, cap):
        self.cap = cap
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.cap:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # TODO


class DQL_agent():

    def __init__(self, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network()
        self.memory = ReplayMemory(100_000)
        # self.optimizer
        self.last_state = np.array()
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        pass

class Deep_q_agent_new:
    """Realize deep Q learning algorithm

        TODO
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        # NEW CODE
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.get_qs(state)
        return np.argmax(act_values[0])

    def get_qs(self, state):
        return self.model.predict(np.array(state)/700)

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        # NEW CODE
        current_states = np.array([transition[0] for transition in mini_batch])/700
        print("current_state: ", current_states)
        current_qs_list = []
        for current_state in current_states:
            current_qs_list.append(self.model.predict(current_state))
        print("current_qs_list: ", current_qs_list)
        new_current_states = np.array([transition[3] for transition in mini_batch])/700
        future_qs_list = []
        for new_current_state in new_current_states:
            future_qs_list.append(self.target_model.predict(new_current_state))
        print("future_q_list: ", future_qs_list)
        X = []
        Y = []

        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            target = reward
            if not done:
              # NEW CODE
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(state)
            Y.append(current_qs)
        self.model.fit(np.array(X)/700, np.array(Y), batch_size=batch_size,
                       verbose=0, shuffle=False)
        if self.target_update_counter > 100:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        if self.epsilon> self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.target_update_counter += 1


class Deep_q_agent:
    """Realize deep Q learning algorithm

        TODO
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9 # discount rate
        self.epsilon = 0.7 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        # NEW CODE
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def get_qs(self, state):
        return self.model.predict(np.array(state))[0]

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            target = reward
            if not done:

                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(np.array(state)/700,steps=1)
                target_f[0][action] = target
                self.model.fit(np.array(state)/700, target_f, epochs=1, verbose=0, shuffle=False)

        if self.epsilon> self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.target_update_counter += 1


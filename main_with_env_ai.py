"""
Project: "Car Simulator"

Autor: Artyom Voronin

"""

import pygame
from math import atan2, cos, sin, sqrt, pi, atan
from math import radians, degrees, copysign
from pygame.math import Vector2
import numpy as np
from numpy import asarray
from numpy import savetxt
import random
import copy
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import time
import os


# Global constants
WIDTH = 1600
HEIGHT = 800
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
MODEL_NAME = 'model_dqn'
MIN_REWARD = -200
EPISODES = 10_000
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
UPDATE_TARGET_EVERY = 5

AGGREGATE_STATS_EVERY = 50 # episodes


# GAME_MODE = input("*********** START ***********\n\
# If you want drive car without race-track [f],\n\
# If you want drive car in race-track [t]: ")

# Initialization of Global properties
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CAR SIMULATOR")
clock = pygame.time.Clock()


class Enviroment():

    ACTION_SPACE_SIZE = 3
    def __init__(self, width, height):
        pygame.init()
        pygame.display.set_caption("Car AI learning")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.my_font = pygame.font.SysFont("monospace", 15)
        self.ticks = 60
        self.done = False
        self.dt = 0


        self.reset_map = False

    def reset(self):
        self.car= car_ai.Car_AI(200,600, 90)
        self.track = map_module.Track()
        self.car.rays.distances(self.track.lines)
        dist_list = self.distances_process(self.car)
        dist_list = np.asarray(dist_list)
        dist_list = dist_list.reshape(1,7)/700
        self.draw_all()

        self.episode_step = 0

        return dist_list

    def draw_all(self):
        self.screen.fill((0, 0, 0))

        label_score = self.my_font.render("Score: {0}".format(self.car.score),1,(0,255,0))
        screen.blit(label_score, (1400, 300))

        label_info = self.my_font.render("For move use [UP,DOWN,LEFT,RIGHT], [q] for exit",1,(0,255,0))
        label_velocity = self.my_font.render("Velocity: {0}".format(int(self.car.velocity.x)),1,(0,255,0))
        self.screen.blit(label_velocity, (1400, 200))
        self.screen.blit(label_info, (1100, 30))
        self.car.update(self.dt)
        self.car.draw()
        self.track.draw()
        pygame.display.update()
        self.car.rays.distance_list = []
        self.clock.tick(self.ticks)



    def step(self, action):
        self.dt = self.clock.get_time() / 1000
        pygame.time.delay(10)
        self.episode_step += 1

        case = np.argmax(action)
        up, down, left, right, space = False,False,False,False,False

        if case == 0:
            left = False
            right = False
        if case == 1:
            left = True
            right = False
        if case == 2:
            left = False
            right = True

        self.car.input_process(self.dt, [up, down, left, right], space)

        col_points = collision_points(self.car.car_body_lines, self.track.lines)
        score_points = collision_points(self.car.car_body_lines, self.track.center_lines)

        for point in score_points:
            if point:
                self.car.score += 25
                if len(self.track.center_lines) == 1:
                    self.car.score +=100
                    self.done = True
                    break
                else:
                    self.track.center_lines.pop(0)
                break

        for i in col_points:
            for j in i:
                if j != None:
                    self.done = True
                    self.car.score -= 300

        self.car.rays.distances(self.track.lines)
        dist_list = self.distances_process(self.car)
        dist_list = np.asarray(dist_list)
        dist_list = dist_list.reshape(1,7)/700
        new_state = dist_list
        reward = self.car.score
        return_values = (new_state, reward, self.done)
#        self.draw_all()
        return return_values

    def distances_process(self, car):
        tmp = []
        for i in car.rays.distance_list:
            if i == None:
                tmp.append(700)
            else:
                tmp.append(i[1])
        return tmp



class Game_player_AI():
    def __init__(self, width, height, model, with_track=True):
        pygame.init()
        pygame.display.set_caption("Car simulator")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.my_font = pygame.font.SysFont("monospace", 15)
        self.ticks = 60
        self.exit = False
        self.reset = False
        self.done = False
        self.with_track = with_track
        self.save_file = False
        self.data_distance = list()
        self.data_buttons = list()
        self.model = model

        self.reset_map = False

    def run(self):
        car=Car_AI(200,600, 90)
        if self.with_track:
            track = Track()

        while not self.exit:
            dt = self.clock.get_time() / 1000
            pygame.time.delay(10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            keys = pygame.key.get_pressed()
            right, left, up, down, space, quit = False, False, False, False, False, False
            if keys[pygame.K_RIGHT]:
                right = True
            if keys[pygame.K_LEFT]:
                left = True
            if keys[pygame.K_UP]:
                up = True
            if keys[pygame.K_DOWN]:
                down = True
            if keys[pygame.K_SPACE]:
                space = True
            if keys[pygame.K_q]:
                self.exit = True

            car.rays.distances(track.lines)
            dist_list = self.distances_process(car)
            dist_list = np.asarray(dist_list)


            predict = self.model.predict(dist_list.reshape(1,7))

            case = np.argmax(predict)

            if case == 0:
                left = False
                right = False
            if case == 1:
                left = True
                right = False
            if case == 2:
                left = False
                right = True

            car.input_process(dt, [up, down, left, right], space)

            # Drawing
            screen.fill((0, 0, 0))

            if self.with_track:
                label_score = self.my_font.render("Score: {0}".format(car.score),1,(0,255,0))
                screen.blit(label_score, (1400, 300))

            label_info = self.my_font.render("For move use [UP,DOWN,LEFT,RIGHT], [q] for exit",1,(0,255,0))
            label_velocity = self.my_font.render("Velocity: {0}".format(int(car.velocity.x)),1,(0,255,0))
            screen.blit(label_velocity, (1400, 200))
            screen.blit(label_info, (1100, 30))
            car.update(dt)
            car.draw()

            if self.with_track:
                track.draw()
                col_points = collision_points(car.car_body_lines, track.lines)
                score_points = collision_points(car.car_body_lines, track.center_lines)

                for point in score_points:
                    if point:
                        car.score += 1000
                        if len(track.center_lines) == 1:
                            track.center_lines.pop(0)
                            self.reset_map = True
                            break
                        else:
                            track.center_lines.pop(0)
                        break

                for i in col_points:
                    for j in i:
                        if j != None:
                            self.reset = True
                            self.exit = True

                car.rays.distances(track.lines)
                dist_list = self.distances_process(car)
                if self.save_file:
                    self.save_data(dist_list, self.button_process([up, down, left, right]))

                if self.exit:
                    if self.save_file:
                        data_dist = asarray(self.data_distance)/700
                        data_butt = asarray(self.data_buttons)
                        savetxt('distance', data_dist, delimiter=',')
                        savetxt('buttons', data_butt, delimiter=',')

            pygame.display.update()
            car.rays.distance_list = []
            self.clock.tick(self.ticks)
            if self.reset_map:
                track.reset()
                self.reset_map = False

    def save_data(self, distance, button):
        self.data_distance.append(distance)
        self.data_buttons.append(button)

    def distances_process(self, car):
        tmp = []
        for i in car.rays.distance_list:
            if i == None:
                tmp.append(700)
            else:
                tmp.append(i[1])
        return tmp

    def button_process(self,buttons):
        up, down, left, right = buttons

        if not left and not right:
            return 0
        if left and not right:
            return 1
        if not left and right:
            return 2

    def predict_process(self, predict):
        # TODO
        pass


class Game_start:
    """Initialize parametrs and start mode

    """
    def __init__(self):
        global GAME_MODE
        self.mode = None
        if GAME_MODE == "f":
            self.mode = False
        if GAME_MODE == "t":
            self.mode = True
        if GAME_MODE == "c":
            pass
        self.model =tf.keras.models.load_model('model.model')
        self.game = Game_player_AI(WIDTH, HEIGHT, self.model, self.mode)

    def run(self):
        while not self.game.reset:
            self.game.run()
            if self.game.reset == False:
                pygame.quit()
                break
            else:
                self.game = Game_player_AI(WIDTH, HEIGHT, self.model, self.mode)



# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self._build_model()

        # Target network
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=7, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        print(model.summary())
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)
        # current_qs_list = []
        # for current_state in current_states:
        #     current_qs_list.append(self.model.predict(current_state))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        # future_qs_list = []
        # for new_current_state in new_current_states:
        #     future_qs_list.append(self.target_model.predict(new_current_state))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            print(action)
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        # callbacks=[self.tensorboard] if terminal_state else None
        print("np.array(X)",np.array(X))
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(1,7))


agent = DQNAgent()
env = Enviroment(WIDTH, HEIGHT)
ep_rewards = [-200]
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
if not os.path.isdir("models"):
    os.makedirs("models")

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)


# if __name__ == '__main__':

 #   game_start = Game_start()
 #    game_start.run()


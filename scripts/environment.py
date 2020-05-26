"""Enviroment wrapper for car_module and map_module.
"""

import pygame
import numpy as np

from scripts.car_module import Car
from scripts.map_module import Track
from scripts.rays import collision_points


class Environment():
    """Class environment.
    Contain all necessary functions for DQN_agent.
    """

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
        self.info = None


        self.reset_map = False

    def reset(self):
        """Reset environment to original state.

        arguments:
            None
        return:
            - cur_state: current state (sensor's distances list)
        """

        self.done = False
        self.car= Car(200, 600, 90)
        self.track = Track(3)
        self.car.rays.distances(self.track.lines, self.screen)
        dist_list = self.distances_process(self.car)
        dist_list = np.asarray(dist_list)
        dist_list = dist_list / 700
        # dist_list = np.append(dist_list, self.car.velocity.x / self.car.max_velocity)

        # self.draw_all() # TODO change draw_all()
        self.car.update(self.dt)
        self.car.rays.distance_list = []

        self.episode_step = 0
        cur_state = dist_list.reshape(1,7)

        # print("[DEBUG-environment.py] cur_state: {}".format(cur_state))
        return cur_state

    def draw_all(self):
        """Draw all stuff :)
        and update to the next.

        arguments:
            None
        return:
            None
        """

        self.screen.fill((0, 0, 0))

        label_score = self.my_font.render("Score: {0}".format(self.car.score),1,(0,255,0))
        self.screen.blit(label_score, (1400, 300))

        label_info = self.my_font.render("For move use [UP,DOWN,LEFT,RIGHT], [q] for exit",1,(0,255,0))
        label_velocity = self.my_font.render("Velocity: {0}".format(int(self.car.velocity.x)),1,(0,255,0))
        self.screen.blit(label_velocity, (1400, 200))
        self.screen.blit(label_info, (1100, 30))
        self.car.draw(self.screen)
        self.track.draw(self.screen)
        pygame.display.update()
        self.clock.tick(self.ticks)



    def step(self, action):
        """Do a step forward.

        arguments:
            - action: action from DQN_agent.
        return:

            - new_state: s_{n+1}.
            - reward: by action on current state s_n.
            - done: if step done or not ???? # TODO.
            - info: some informations about episode.

        """
        reward = 0
        self.dt = self.clock.get_time() / 1000
        pygame.time.delay(10)
        self.episode_step += 1

        # case = np.argmax(action)
        case = action
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

        self.car.input_process_ai(self.dt, [up, down, left, right], space)

        col_points = collision_points(self.car.car_body_lines, self.track.lines)
        score_points = collision_points(self.car.car_body_lines, self.track.center_lines)

        for point in score_points:
            if point:
                reward += 300
                self.car.score += 300
                if len(self.track.center_lines) == 1:
                    reward += 10000
                    self.car.score += 10000
                    # self.done = True
                    # self.track.reset()
                    break
                else:
                    self.track.center_lines.pop(0)
                break

        for i in col_points:
            for j in i:
                if j != None:
                    self.done = True
                    reward -= 1000
                    self.car.score -= 1000
                    break

        self.car.rays.distances(self.track.lines, self.screen)
        dist_list = self.distances_process(self.car)
        dist_list = np.asarray(dist_list)
        new_state = (dist_list / 700).reshape(1,7)
        # new_state = np.append(dist_list,
        #        self.car.velocity.x / self.car.max_velocity).reshape(1,8)
        # print("[DEBUG-environment.py] new_state: {}".format(dist_list))
        self.car.score += int(self.episode_step * 0.1)
        reward += int(self.episode_step* 0.1)
        # reward = self.car.score
        return_values = (new_state, reward, self.done, self.info)
        # self.draw_all() # TODO draw_all() from train.py

        self.car.update(self.dt)
        self.car.rays.distance_list = []

        return return_values

    def distances_process(self, car):
        """Process the distance between car and walls.

        arguments:
            - car: object car.
        return:
            tmp: processed list of distances.
        """
        tmp = []
        for i in car.rays.distance_list:
            if i == None:
                tmp.append(700)
            else:
                tmp.append(i[1])
        return tmp



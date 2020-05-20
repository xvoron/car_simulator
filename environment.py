"""
Enviroment architecture


"""

import pygame
import car_module
import map_module
import numpy as np


class Environment():

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
        self.car= car_module.Car(200,600, 90)
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

        self.car.input_process_ai(self.dt, [up, down, left, right], space)

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



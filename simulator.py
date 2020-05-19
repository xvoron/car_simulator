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
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

# Global constants
WIDTH = 1600
HEIGHT = 800

GAME_MODE = input("*********** START ***********\n\
If you want drive car without race-track [f],\n\
If you want drive car in race-track [t]: ")

# Initialization of Global properties
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CAR SIMULATOR")
clock = pygame.time.Clock()


# Global functions

def collision_points(car_lines, map_lines):
    """Calculate collision points in two arrays of lines

    arg: car_lines; type: array -> body car lines
    arg: map_lines; type: array -> lines create track or some another

    return: collision_points; type: array -> points where lines collision
    """
    collision_points = []
    for map_line in map_lines:
        for car_line in car_lines:
            collision_2_lines_responde = collision_2_lines(map_line, car_line)
            if collision_2_lines_responde is not None:
                collision_points.append(collision_2_lines_responde)
    return collision_points


def collision_2_lines(l1, l2):
    """Calculate collision point in two line

    arg: l1; type: array -> line # 1
    arg: l2; type: array -> line # 2

    return: [Px, Py]; type: list:float -> point where lines collision

    more info: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """
    x1, y1, x2, y2 = l1[0][0], l1[0][1], l1[1][0], l1[1][1]
    x3, y3, x4, y4 = l2[0][0], l2[0][1], l2[1][0], l2[1][1]
    den =  ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    if den == 0:
        return
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/den
    u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/den
    if t>=0 and t<=1 and u>=0 and u<=1:
        Px =(x1 + t*(x2-x1))
        Py =(y1 + t*(y2-y1))
        return [Px, Py]


def draw_rect(center, corners, rotation_angle, color):
    """Rotate and draw rectangle

    arg: center; type: array -> center of rectangle
    arg: corners; type: array -> corners of rectangle
    arg: rotation_angle; type: int:degrees -> angle of rotation
    arg: color; type: tuple -> color of rectangle

    return: rotated_corners; type: list:float -> new rotated corners
    """
    c_x = center[0]
    c_y = center[1]
    delta_angle = radians(rotation_angle)
    rotated_corners = []
    for p in corners:
        temp = []
        length = sqrt((p[0] - c_x)**2 + (c_y - p[1])**2)
        angle = atan2(c_y - p[1], p[0] - c_x)
        angle += delta_angle
        temp.append(c_x + length*cos(angle))
        temp.append(c_y - length*sin(angle))
        rotated_corners.append(temp)

    # draw rectangular polygon --> car body
    rect = pygame.draw.polygon(screen, color,
                               (rotated_corners[0], rotated_corners[1],
                                rotated_corners[2], rotated_corners[3]))
    return rotated_corners


class Ray:
    """Class represent geometrical rays

    Needed to calculate distance between 2 objects
    """
    def __init__(self, position, angle, ID):
        self.angle = radians(angle)
        self.pos = [position[0], position[1]]
        self.ID = ID
        self.distance = 0
        self.ray_length = 300
        self.end_point = [self.ray_length * cos(self.angle) + self.pos[0],
                          self.pos[1] - self.ray_length * sin(self.angle)]
        self.ray_line = [[self.pos, self.end_point]]
        # self.intersection_line = [[self.pos, self.end_point]]

    def update(self, position, angle):
        self.angle = radians(angle)
        self.pos = [position[0], position[1]]
        self.end_point = [self.ray_length * cos(self.angle) + self.pos[0],
                          self.pos[1] - self.ray_length * sin(self.angle)]
        self.ray_line = [[self.pos, self.end_point]]

    def is_any_intersection(self, lines):
        points = collision_points(self.ray_line, lines)
        for point in points:
            if point:
                self.distance = self.calculate_distance(point)
                pygame.draw.circle(screen, (255, 0, 0),
                                   (int(point[0]), int(point[1])), 4)
                # self.intersection_line = [[self.pos],[point]]
                return [self.ID, self.distance]
            else:
                self.distance = None
                return [self.ID, self.distance]

    def calculate_distance(self, point):
        return sqrt((point[0]-self.pos[0])**2 + (point[1]-self.pos[1])**2)

    def draw(self):
        pygame.draw.line(screen, (0, 0, 0),
                         self.ray_line[0][0],
                         self.ray_line[0][1])

    def draw_intersection(self):
        pass



class Rays:
    """Class representing high level comunication with class Ray

    Needed to create bench of lines that represent sensors.
    """
    def __init__(self, position, angle):
        self.ray_list = []
        self.distance_list = []
        tmp = 0
        for i in range(7):
            self.ray_list.append(Ray(position, angle-90+tmp, i))
            tmp += 30

    def update(self, position, angle):
        tmp = 0
        for ray in self.ray_list:
            ray.update(position, angle - 90 + tmp)
            tmp += 30

    def draw(self):
        for ray in self.ray_list:
            ray.draw()

    def distances(self, lines):
        for ray in self.ray_list:
            self.distance_list.append(ray.is_any_intersection(lines))


class Track:
    """Class represent a race track
    """
    def __init__(self):
        self.points_map_out = [[100,700], [100,500], [100,400], [200,200],
                               [300,100], [400,100], [600,200], [700,200],
                               [800,100], [1100,100], [1300,200], [1300,500],
                               [1200,600], [1100,600], [1100,700], [100,700]]

        self.points_map_in =  [[300,600], [300,500], [400,400], [400,300],
                               [700,300], [900,200], [1100,200], [1200,300],
                               [1200,400], [1100,400], [900,600], [300,600]]

        self.center_lines = [[[100,500],[300,500]],[[100,450],[350,450]],
                             [[100,400],[400,400]],[[150,300],[400,350]],
                             [[200,200],[400,300]],[[300,100],[500,300]],
                             [[600,200],[700,300]],
                             [[800,100],[900,200]], [[1000,100],[1000,200]],
                             [[1100,100],[1100,200]],
                             [[1300,200],[1200,300]], [[1200,400],[1300,500]],
                             [[900,600],[1100,600]], [[700,600],[700,700]],
                             [[500,600],[500,700]]]


        # Create lines from points
        self.lines = []
        for i in range(len(self.points_map_out)):
            if i == 0:
                self.lines.append([self.points_map_out[-1],self.points_map_out[0]])
            else:
                self.lines.append([self.points_map_out[i-1],self.points_map_out[i]])
        for i in range(len(self.points_map_in)):
            if i == 0:
                self.lines.append([self.points_map_in[-1],self.points_map_in[0]])
            else:
                self.lines.append([self.points_map_in[i-1],self.points_map_in[i]])

        self.lines_original = copy.copy(self.center_lines)

    def draw(self):
        global screen
        pygame.draw.lines(screen, (0, 0, 255), False, self.points_map_out)
        pygame.draw.lines(screen, (0, 0, 255), False, self.points_map_in)
        for line in self.center_lines:
            pygame.draw.line(screen, (0 , 255, 0), line[0], line[1])

    def reset(self):
        self.center_lines = self.lines_original[:len(self.lines_original)]


class Car:
    """Class represent a car

    Do car physics and comunicate with Rays class
    """
    def __init__(self, x, y, angle=0.0, length=50,
                 max_steering=30, max_acceleration=30.0):

        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle

        self.car_width = 50
        self.car_height = 100
        self.length = length

        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 300
        self.brake_deceleration = 1000
        self.free_deceleration = 5

        self.acceleration = 0.0
        self.steering = 0.0
        self.turning_radius_circle = 0.0
        self.car_body_lines = None

        self.score = 0
        self.rays = Rays(self.position, self.angle)

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity,
                              min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            self.turning_radius_circle = turning_radius

            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt
        self.rays.update(self.position, self.angle)

    def draw(self):
        global screen
        self.rays.draw()
        self.draw_ackermann()
        rotated_corners =draw_rect(self.position,
                                  [[self.position[0]-6, self.position[1]-15],
                                  [self.position[0]+54, self.position[1]-15],
                                  [self.position[0]+54, self.position[1]+15],
                                  [self.position[0]- 6, self.position[1]+15]],
                                  self.angle, (255,0,0))
        rc= rotated_corners
        self.car_body_lines = [[rc[1], rc[0]], [rc[2], rc[1]],
                               [rc[3], rc[2]], [rc[0], rc[3]]]

        # self.rays.draw()

    def ray_distances(self,lines):
        self.rays.distances(lines)

    def input_process(self,dt, action, space=False):
        up, down, left, right = action[0], action[1], action[2], action[3],

        if left:
            self.steering += 30 * dt
        elif right:
            self.steering -= 30 * dt
        else:
            self.steering = 0
        self.steering = max( -self.max_steering,
                            min(self.steering, self.max_steering))

        if up:
            if self.velocity.x < 0:
                self.acceleration = self.brake_deceleration
            else:
                self.acceleration += 1000 * dt
        elif down:
            if self.velocity.x > 0:
                self.acceleration = -self.brake_deceleration
            else:
                self.acceleration -= 1000 * dt
        elif space:
            if abs(self.velocity.x) > dt * self.brake_deceleration:
                self.acceleration = - copysign(self.brake_deceleration,
                                               self.velocity.x)
            else:
                self.acceleration = -self.velocity.x / dt
        else:
            if abs(self.velocity.x) > dt * self.free_deceleration:
                self.acceleration = - copysign(self.free_deceleration,
                                               self.velocity.x)
            else:
                if dt != 0:
                    self.acceleration = -self.velocity.x / dt

        self.acceleration = max(-self.max_acceleration,
                                min(self.acceleration, self.max_acceleration))

    def draw_ackermann(self):
        color = (0,255,0)
        l = 60
        w = 45
        fi = radians(self.steering)
        center_1_wheel = [1350, 700]
        center_2_wheel = [1500, 700]
        fi_i = degrees(atan((2*l*sin(fi))/(2*l*cos(fi)-w*sin(fi))))
        fi_o = degrees(atan((2*l*sin(fi))/(2*l*cos(fi)+w*sin(fi))))

        corners1 = [[center_1_wheel[0]-15,center_1_wheel[1]-30],
                    [center_1_wheel[0]+15,center_1_wheel[1]-30],
                    [center_1_wheel[0]+15,center_1_wheel[1]+30],
                    [center_1_wheel[0]-15,center_1_wheel[1]+30]]

        corners2 = [[center_2_wheel[0]-15,center_2_wheel[1]-30],
                    [center_2_wheel[0]+15,center_2_wheel[1]-30],
                    [center_2_wheel[0]+15,center_2_wheel[1]+30],
                    [center_2_wheel[0]-15,center_2_wheel[1]+30]]

        draw_rect(center_1_wheel,corners1, fi_i, color)
        draw_rect(center_2_wheel,corners2, fi_o, color)
        pygame.draw.line(screen, color, center_1_wheel, center_2_wheel)



class Game_player():
    def __init__(self, width, height, with_track=True):
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
        self.data_distance = np.array([])
        self.data_buttons = np.array([])

    def run(self):
        car=Car(200,600, 90)
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
                            self.reset = True
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
                if self.save_file:
                    self.save_data(car.rays.distances, self.button_process([up, down, left, right]))

            pygame.display.update()
            car.rays.distance_list = []
            self.clock.tick(self.ticks)
            if self.exit:
                if self.save_file:
                    data_dist = asarray(self.data_distance)
                    data_butt = asarray(self.data_buttons)
                    savetxt('distance', data_dist, delimiter=',')
                    savetxt('buttons', data_butt, delimiter=',')

    def save_data(self, distance, button):
        np.append(self.data_distance, distance, axis=0)
        np.append(self.data_buttons, button, axis=0)

    def button_process(self,buttons):
        up, down, left, right = buttons

        if not up and not down and not left and not right:
            return 0
        if up and not down and not left and not right:
            return 1
        if not up and down and not left and not right:
            return 2
        if not up and not down and left and not right:
            return 3
        if not up and not down and not left and right:
            return 4
        if up and not down and left and not right:
            return 5
        if up and not down and not left and right:
            return 6
        if not up and down and left and not right:
            return 7
        if not up and down and not left and right:
            return 8


class Game_start:
    """Initialize parametrs and start mode

    """
    def __init__(self):
        global GAME_MODE
        mode = None
        if GAME_MODE == "f":
            mode = False
        if GAME_MODE == "t":
            mode = True
        if GAME_MODE == "c":
            pass
        self.game = Game_player(WIDTH, HEIGHT, mode)

    def run(self):

        while not self.game.reset:
            self.game.run()
            if self.game.reset == False:
                pygame.quit()
                break
            else:
                self.game = Game_player(WIDTH, HEIGHT)




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


if __name__ == '__main__':

    game_start = Game_start()
    game_start.run()


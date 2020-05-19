"""Car module

Create and provide car kinematic simulation.

"""

import rays
import pygame
from pygame.math import Vector2
from math import atan2, cos, sin, sqrt, pi, atan
from math import radians, degrees, copysign

def draw_rect(center, corners, rotation_angle, color, screen):
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
        self.rays = rays.Rays(self.position, self.angle)

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


    def ray_distances(self,lines, screen):
        self.rays.distances(lines, screen)

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

    def input_process_ai(self, dt, action, space=False):
        up, down, left, right = action[0], action[1], action[2], action[3],

        if left:
            self.steering += 30 * dt
        elif right:
            self.steering -= 30 * dt
        else:
            self.steering = 0
        self.steering = max( -self.max_steering,
                            min(self.steering, self.max_steering))

        self.acceleration += 1000 * dt

        self.acceleration = max(-self.max_acceleration,
                                min(self.acceleration, self.max_acceleration))

    def draw(self, screen):
        self.rays.draw(screen)
        self.draw_ackermann(screen)
        rotated_corners =draw_rect(self.position,
                                  [[self.position[0]-6, self.position[1]-15],
                                  [self.position[0]+54, self.position[1]-15],
                                  [self.position[0]+54, self.position[1]+15],
                                  [self.position[0]- 6, self.position[1]+15]],
                                  self.angle, (255,0,0), screen)
        rc= rotated_corners
        self.car_body_lines = [[rc[1], rc[0]], [rc[2], rc[1]],
                               [rc[3], rc[2]], [rc[0], rc[3]]]

        # self.rays.draw()

    def draw_ackermann(self, screen):
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

        draw_rect(center_1_wheel,corners1, fi_i, color, screen)
        draw_rect(center_2_wheel,corners2, fi_o, color, screen)
        pygame.draw.line(screen, color, center_1_wheel, center_2_wheel)



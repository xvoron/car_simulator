"""Rays module.

Create and manipulate with geometrical rays.
In this case rays are sensors that mesuare a distance
from car to a wall.


    \   |   /
     \  |  /   <---- Rays
      \ | /
      +---+
      |   |
      |   |    <---- Car
      |   |
      +---+

"""
# Imports
import pygame

from math import atan2, cos, sin, sqrt, pi, atan
from math import radians, degrees, copysign

# Global functions
SENSORS_NUMBER = 7

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
    # TODO more than one colission
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

    def is_any_intersection(self, lines, screen):
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

    def draw(self, screen):
        pygame.draw.line(screen, (0, 0, 0),
                         self.ray_line[0][0],
                         self.ray_line[0][1])

    def draw_intersection(self, screen):
        pass



class Rays:
    """Class representing high level comunication with class Ray

    Needed to create bench of lines that represent sensors.
    """
    def __init__(self, position, angle):
        self.ray_list = []
        self.distance_list = []
        tmp = 0
        for i in range(SENSORS_NUMBER):
            self.ray_list.append(Ray(position, angle-90+tmp, i))
            tmp += 30

    def update(self, position, angle):
        tmp = 0
        for ray in self.ray_list:
            ray.update(position, angle - 90 + tmp)
            tmp += 30

    def draw(self, screen):
        for ray in self.ray_list:
            ray.draw(screen)

    def distances(self, lines, screen):
        for ray in self.ray_list:
            self.distance_list.append(ray.is_any_intersection(lines, screen))



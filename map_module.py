"""Map module

Create a race-map/track, where car has been driven.

"""
import copy
import pygame

class Track:
    # TODO Problem when car take all revard lines. There is no reset!

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

    def draw(self, screen):
        pygame.draw.lines(screen, (0, 0, 255), False, self.points_map_out)
        pygame.draw.lines(screen, (0, 0, 255), False, self.points_map_in)
        for line in self.center_lines:
            pygame.draw.line(screen, (0 , 255, 0), line[0], line[1])

    def reset(self):
        # TODO
        self.center_lines = self.lines_original[:len(self.lines_original)]





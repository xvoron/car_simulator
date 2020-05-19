"""Main Car-AI-simulation project file


"""

import pygame

import ai
import car_module
import map_module
import numpy as np
from rays import collision_points

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
        car= car_module.Car(200,600, 90)
        if self.with_track:
            track = map_module.Track()

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
            car.draw(screen)
            if self.with_track:
                track.draw(screen)
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

                car.rays.distances(track.lines, screen)
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


if __name__ == '__main__':

    game_start = Game_start()
    game_start.run()


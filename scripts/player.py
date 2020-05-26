import pygame
import scripts.car_module
import scripts.map_module
import numpy as np
from scripts.rays import collision_points
WIDTH = 1600
HEIGHT = 800
global GLOBAL_MODE


class Game_player():

    def __init__(self, width, height, mode):
        global GLOBAL_MODE
        GLOBAL_MODE = mode
        pygame.init()
        pygame.display.set_caption("Car simulator")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.my_font = pygame.font.SysFont("monospace", 15)
        self.ticks = 60

        if mode == "free":
            self.with_track = False
        elif mode == "race":
            self.with_track = True
        elif mode == "train":
            self.with_track = True
            self.train_mode = True
        elif mode == "ai_mode":
            self.with_track == True
            self.ai_mode = True


        self.exit = False
        self.reset = False
        self.done = False

    def run(self):

        while not self.reset:

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

                self.screen.fill((0, 0, 0))

                if self.with_track:
                    label_score = self.my_font.render("Score: {0}".format(car.score),1,(0,255,0))
                    self.screen.blit(label_score, (1400, 300))
                label_info = self.my_font.render("For move use [UP,DOWN,LEFT,RIGHT], [q] for exit",1,(0,255,0))
                label_velocity = self.my_font.render("Velocity: {0}".format(int(car.velocity.x)),1,(0,255,0))
                self.screen.blit(label_velocity, (1400, 200))
                self.screen.blit(label_info, (1100, 30))
                car.update(dt)
                car.draw(self.screen)
                if self.with_track:
                    track.draw(self.screen)
                    col_points = collision_points(car.car_body_lines, track.lines)
                    score_points = collision_points(car.car_body_lines, track.center_lines)
                    for point in score_points:
                        if point:
                            car.score += 1000
                            if len(track.center_lines) == 1:
                                track.center_lines.pop(0)
                                self.reset = True
                                # track.reset()
                                break
                            else:
                                track.center_lines.pop(0)
                            break

                    for i in col_points:
                        for j in i:
                            if j != None:
                                self.reset = True
                                self.exit = True

                    car.rays.distances(track.lines, self.screen)

                pygame.display.update()
                car.rays.distance_list = []
                self.clock.tick(self.ticks)

            if self.reset == False:
                pygame.quit()
                break
            else:
                # self.game = Game_player(WIDTH, HEIGHT)
                self.__init__(WIDTH, HEIGHT, GLOBAL_MODE)
                # continue



if __name__ == "__main__":
    WIDTH = 1600
    HEIGHT = 800
    Game_player(WIDTH, HEIGHT, "free").run()

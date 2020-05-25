import pygame
import car_module
import map_module
import numpy as np
from rays import collision_points
from environment import Environment
from ai import DQN_agent
WIDTH = 1600
HEIGHT = 800

class AI_vs_Player():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car simulator")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.my_font = pygame.font.SysFont("monospace", 15)
        self.ticks = 60


        self.exit = False
        self.reset = False
        self.done_ai = False
        self.done_player = False

        self.with_track = True

        self.car_ai = car_module.Car(200,600, 90, color=(255,0,0))
        self.car_player = car_module.Car(200, 600, 90, color=(255,255,255))
        self.dqn_agent = DQN_agent("success.model")

        self.track = map_module.Track()

    def run(self):

        while  not self.reset:

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

                self.car_player.input_process(dt, [up, down, left, right], space)


                self.car_ai.rays.distances(self.track.lines, self.screen)
                cur_state = self.distances_process(self.car_ai)
                cur_state = np.asarray(cur_state)
                cur_state = (cur_state / 700).reshape(1,7)
                case = self.dqn_agent.act(cur_state)

                up, down, left, right, space = False, False, False, False, False

                if case == 0:
                    left = False
                    right = False
                if case == 1:
                    left = True
                    right = False
                if case == 2:
                    left = False
                    right = True

                self.car_ai.input_process_ai(dt, [up, down, left, right], space)
                self.car_ai.rays.distance_list = []

                # TODO car_ai

                self.screen.fill((0, 0, 0))

                if self.with_track:
                    label_score = self.my_font.render("Scores: player: {0}, ai: \
                            {1}".format(self.car_player.score,self.car_ai.score),1,(0,255,0))
                    self.screen.blit(label_score, (1400, 300))
                    label_info = self.my_font.render("For move use [UP,DOWN,LEFT,RIGHT], [q] for exit",1,(0,255,0))
                    self.screen.blit(label_info, (1100, 30))
                    if not self.done_player:
                        self.car_player.update(dt)
                        self.car_player.draw(self.screen)
                    if not self.done_ai:
                        self.car_ai.update(dt)
                        self.car_ai.draw(self.screen)

                    if self.with_track:
                        self.track.draw(self.screen)
                        col_points_player = \
                        collision_points(self.car_player.car_body_lines,
                                self.track.lines)
                        col_points_ai = \
                        collision_points(self.car_ai.car_body_lines, self.track.lines)

                        for i in col_points_player:
                            for j in i:
                                if j != None:
                                    self.done_player = True
                                    self.reset = True


                        for i in col_points_ai:
                            for j in i:
                                if j != None:
                                    self.done_ai = True
                                    self.reset = True


                pygame.display.update()
                self.car_ai.rays.distance_list = []
                self.car_player.rays.distance_list = []
                self.clock.tick(self.ticks)
                if self.done_ai and not self.done_player:
                    print("================= Player win! =================")
                elif not self.done_ai and self.done_player:
                    print("=================== AI win! ===================")




                if  self.exit:
                    pygame.quit()
                    break
                elif self.reset:
                    # self.game = Game_player(WIDTH, HEIGHT)
                    self.__init__()
                    # continue



    def distances_process(self, car):
        tmp = []
        for i in car.rays.distance_list:
            if i == None:
                tmp.append(700)
            else:
                tmp.append(i[1])
        return tmp


if __name__ == "__main__":
    AI_vs_Player().run()

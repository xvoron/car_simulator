"""Main Car-AI-simulation project file

"""

import pygame

import ai
import car_module
import map_module
import numpy as np
from rays import collision_points
from environment import Environment
from player import Game_player

# Global constants
WIDTH = 1600
HEIGHT = 800

GAME_MODE = input("===================== START ======================\n\
If you want drive car without race-track ----- [f]\n\
If you want drive car in race-track ---------- [r]\n\
Train model ---------------------------------- [t]\n\
Compete with AI ------------------------------ [c]\n\
> ")

# Initialization of Global properties
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CAR SIMULATOR")
clock = pygame.time.Clock()


class Game_start:
    """Initialize parametrs and start mode

    """
    def __init__(self):
        global GAME_MODE
        if GAME_MODE == "f":
            mode = "free"
        elif GAME_MODE == "r":
            mode = "race"
        elif GAME_MODE == "t":
            mode = "train"
            # TODO
        elif GAME_MODE == "c":
            mode = "compete"
            # TODO
        else:
            mode = "f"

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


"""Main Car-AI-simulation project file

"""

import pygame

import ai
import car_module
import map_module
import numpy as np
from environment import Environment
from player import Game_player
from train import train, ai_race
from ai_vs_player import AI_vs_Player

# Global constants
WIDTH = 1600
HEIGHT = 800

GAME_MODE = input("===================== START ======================\n\
If you want drive car without race-track ----- [f]\n\
If you want drive car in race-track ---------- [r]\n\
Train model ---------------------------------- [t]\n\
AI race with 'succses.model' ----------------- [a]\n\
Compete with AI #TODO ------------------------ [c]\n\
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
            Game_player(WIDTH, HEIGHT, mode).run()
        elif GAME_MODE == "r":
            mode = "race"
            Game_player(WIDTH, HEIGHT, mode).run()
        elif GAME_MODE == "t":
            mode = "train"
            train()
        elif GAME_MODE == "a":
            mode = "ai_mode"
            ai_race()

            # TODO
        elif GAME_MODE == "c":
            mode = "compete"
            AI_vs_Player().run()
            # TODO
        else:
            mode = "f"
            Game_player(WIDTH, HEIGHT, mode).run()


if __name__ == '__main__':

    Game_start()


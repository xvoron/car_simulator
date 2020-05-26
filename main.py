"""Main Car-AI-simulation project file

This project is my attempt to learn Pygames library
and to understand basics of Deep Q-learning algorithm.

https://www.pygame.org/news

@author: Artyom Voronin
https://github.com/xvoron/

"""

from scripts.player import Game_player
from scripts.train import train, ai_race
from scripts.ai_vs_player import AI_vs_Player

# Global constants
WIDTH = 1600
HEIGHT = 800

def start():
    """Chose a game mode and start
    """

    GAME_MODE = input("===================== START ======================\n\
    If you want drive car without race-track ----- [f]\n\
    If you want drive car in race-track ---------- [r]\n\
    Train model ---------------------------------- [t]\n\
    AI race with 'succses.model' ----------------- [a]\n\
    Compete with AI #TODO ------------------------ [c]\n\
    > ")

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

    elif GAME_MODE == "c":
        mode = "compete"
        AI_vs_Player().run() # TODO debug ai_vs_player.py
    else:
        mode = "f"
        Game_player(WIDTH, HEIGHT, mode).run()

if __name__ == '__main__':
    start()


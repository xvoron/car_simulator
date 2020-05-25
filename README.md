# Deep Q-learning agent for 2D car simulator

This project is my attempt to learn Pygame library and to understand
basics of Deep Q-learning algorithm.

# TODO

  - \[x\] Environment
  - <span class="done2"></span>refactoring
  - <span class="done0"></span>solve \# TODO

# Using

``` 
    $python main.py

    If you want drive car without race-track --- [f] 
    If you want drive car in race-track -------- [r]
    Train model -------------------------------- [t]
    Competitive with AI ------------------------ [c]
    >
```

# Files

  - ai.py:
      - class Nework - create a network.
      - class ReplayMemory - create a memory from where we will take a
        samples.
      - DQL\\\_agent - main part of the algorithm

<!-- end list -->

  - car\_module.py
  - environment.py
      - class Environment
  - main.py
      - class Game\_start
  - map\_module.py
      - class Track
  - rays.py:
      - class Ray - class to manipulate with one geometrical Ray
      - class Rays - create and manipulate with bunch of rays (wraper
      - for class Ray)
  - simulator.py: all in one. Old file.

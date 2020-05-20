= Deep Q-learning agent for 2D car simulator = 

This project is my attempt to learn Pygame library and to understand
basics of Deep Q-learning algorithm. 

= TODO =

* [x] Environment
* [o] refactoring
* [ ] Solve # TODO 


= Using =
{{{

$python main.py

If you want drive car without race-track --- [f] 
If you want drive car in race-track -------- [r]
Train model -------------------------------- [t]
Competitive with AI ------------------------ [c]
>

}}}

= Files =
- ai.py:
    - class Nework - create a network.
    - class ReplayMemory - create a memory from where we will take a samples.
    - DQL_agent - main part of the algorithm

- car_module.py
    - class Car
        - for ai ------- *input_process_ai()*
        - for player --- *input_process()*
- environment.py
    - class Environment 
- main.py
    - class Game_start 
- map_module.py
    - class Track 
- rays.py:
    - class Ray - class to manipulate with one geometrical Ray 
    - class Rays - create and manipulate with bunch of rays (wraper for class
      Ray)
      
- simulator.py: all in one. Old file. 






# Deep Q-learning agent for 2D car simulator

This project is my attempt to learn Pygame library and to understand
basics of Deep Q-learning algorithm.

## Installation

    $cat requirements.txt
    pygame
    numpy
    keras
    collections
    copy
    random
    math
    
    $pip install -r requirements.txt

## Usage

``` 
    $python main.py

    If you want drive car without race-track --- [f] 
    If you want drive car in race-track -------- [r]
    Train model -------------------------------- [t]
    AI drive car with 'succses.model' ---------- [a]
    Competitive with AI ------------------------ [c]
    >
```

## Files

  - script/
      - ai.py: DQL\_agent - main part of the algorithm.
      - car\_module.py: Car dynamics and engine.
      - environment.py: Environment for DQL\_agent usage.
      - map\_module.py: Race-track.
      - rays.py: Geometrical rays implementation (Lidar for car).
      - train.py: Training model functions.
      - player.py: Player logic module.
      - ai\_vs\_player.py: Pre-trained model 'success.model' vs player.
  - main.py: Main script.
  - success.model: Model after 10000 iterations

## TODO

  - <span class="done0"></span>Debug ai\_vs\_player.py

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Car_rl-v0',
    entry_point='gym_soccer.envs:SoccerEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)

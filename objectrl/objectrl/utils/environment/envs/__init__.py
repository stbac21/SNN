
# import envs and necessary gym packages
from objectrl.utils.environment.envs.simple_env import SimpleWalkerEnvClass
from gymnasium.envs.registration import register

# register the env using gym's interface
register(
    id = 'SimpleWalkingEnv-v0',
    entry_point = 'objectrl.utils.environment.envs.simple_env:SimpleWalkerEnvClass', # FIXME - surely there's just no way its gotta be this explicit, right?
    max_episode_steps = 500
)


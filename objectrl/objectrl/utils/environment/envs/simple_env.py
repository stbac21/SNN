import random
from gymnasium import spaces
from evogym import EvoWorld, sample_robot
from evogym.envs import EvoGymBase

from typing import Optional, Dict, Any, Tuple
import numpy as np
import os

class SimpleWalkerEnvClass(EvoGymBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):
        # make world
        #self.world = EvoWorld.from_json(os.path.abspath('objectrl/utils/environment/world_data/Test.json'))    # Test.json has 'robot' object included
        # body, connections = sample_robot((5,5))
        
        self.world = EvoWorld.from_json(os.path.abspath('objectrl/utils/environment/world_data/Walker-v0.json'))
        #self.world.add_from_array('robot', body, random.randint(1,5), random.randint(1,5), connections=connections) # not sure if this works as intended
        self.world.add_from_array('robot', body, 2, 2, connections=connections)
        
        # init sim
        EvoGymBase.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        obs_size = self.reset()[0].size

        # NOTE: "low= " CONSTRAINS COMPRESSIBILITY OF VOXELS; "high= " CONSTRAINS EXPANDABILITY; TOGETHER THEY DECIDE INTERVAL SIZE OF ACTION SPACE.
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float) # MIGHT NEED TO BE ADJUSTED - FIXME
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_size,), dtype=float)

        # set viewer to track objects
        self.default_viewer.track_objects('robot')

    def step(self, action):
        # collect pre step information - NOTE: FOR ROBOT MOVEMENT REWARD; FOR WALKER, USE ONLY THIS; FOR OTHER TASKS, CONSIDER ADDITIONAL/ALTERNATIVE REWARD SCHEMES.
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])

        # NOTE: ALTERNATIVE REWARD SCHEMES ARE A "DOWN-THE-ROAD" CONSIDERATION FOR NON-EASY TASKS; SO LET'S JUST START WITH GETTING STUFF TO WORK AND TO LEARN SOMETHING PLEASE
        ## collect pre step information - NOTE: FOR BOULDER DISTANCE REWARD; FOR THROWER, REMOVE ROBOT MOVEMENT REWARD; FOR PUSHER, USE BOTH.
        #pos_1 = self.object_pos_at_time(self.get_time(), "boulder")
        ## step
        #done = super().step({'robot': action})
        ## collect post step information
        #pos_2 = self.object_pos_at_time(self.get_time(), "blouder")
        ## compute reward
        #com_1 = np.mean(pos_1, 1)
        #com_2 = np.mean(pos_2, 1)
        #reward = (com_2[0] - com_1[0])

        # NOTE: FOR CATCHER, REWARD SHOULD BE BASED ON Y-POSITION OF BOULDER BEING > 0 DISTANCE FROM THE GROUND; ROBOT POSITION SHOULD BE UNDER BOULDER, ANYTHING ELSE SHOULD BE NEGATIVE REWARD.
        # (SAME THING KINDA APPLIES TO BALANCER; ROBOT JUST NEEDS TO HAVE Y-POSITION DISTANCE TO THE GROUND AT ALL TIMES OR ELSE BIG NEGATIVE REWARD.)

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING") # wording might be a bit extreme but idk what causes it so maybe its representative
            reward -= 3.0
            
        # check goal met
        if com_2[0] > 28:
            done = True
            reward += 1.0

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))
        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs, {}
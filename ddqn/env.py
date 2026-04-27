import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(old_shape[2], old_shape[0], old_shape[1]), 
            dtype=np.uint8
        )

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32)

class MinigridDoorKeyFullyObs(gym.Wrapper):
    def __init__(self, size, render=False):
        env_name = f'MiniGrid-DoorKey-{size}x{size}-v0'
        if render:
            env = gym.make(env_name, max_steps=1000)
        else:
            env = gym.make(env_name,max_steps = 1000)
            
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = ChannelFirst(env)
        env = ScaledFloatFrame(env)
        
        super().__init__(env)
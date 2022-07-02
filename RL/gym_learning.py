# following the tutorial on: https://blog.paperspace.com/getting-started-with-openai-gym/

import gym
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space

print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

# =======
obs = env.reset()
print(f"The initial observation is {obs}")

# sampling a random action
random_action = env.action_space.sample()

# take action
new_obs, reward, done, info = env.step(random_action)
print(f"The new observation is {new_obs}")

# display on the screen
env_screen = env.render(mode="rgb_array")
plt.imshow(env_screen)
# or
#env.render(mode="human")
#import time
#time.sleep(5)
#env.close()

# === Sample run
import time

num_steps = 1
obs = env.reset()

for step in range(num_steps):
	action = env.action_space.sample()
	obs, reward, done, info = env.step(action)
	env.render(mode="human")
	time.sleep(0.001)
	if done:
		env.reset()
env.close()


# ========== SPACES
print(f"Upper bound for env observation: {env.observation_space.high}")
print(f"Lower bound for env observation: {env.observation_space.low}")

# ======== WRAPPER
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs = env.reset()

num_steps = 100
for i in range(num_steps):
	action = env.action_space.sample()
	obs, reward, done, info = env.step(action)
	#env.render()
	time.sleep(0.01)
env.close()

# -
from collections import deque
from gym import spaces
import numpy as np

class ConcatObs(gym.Wrapper):
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		# k is the number of past frames to concatenate
		self.k = k		
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = \
			spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)


	def reset(self):
		ob = self.env.reset()
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()
	
	def _get_ob(self):
		return np.array(self.frames)
	
	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info


env = gym.make("BreakoutNoFrameskip-v4")
wrapped_env = ConcatObs(env, 4)
print(f"The new observation space is: {wrapped_env.observation_space}")
			
obs = wrapped_env.reset()
print(f"Initial observation is of the shape {obs.shape}")
obs, reward, done, info = wrapped_env.step(2)
print(f"Observation after taking a step is: {obs.shape}")

# Another Wrapper demonstration:
import random

class ObservationWrapper(gym.ObservationWrapper):
	def __init__(self, env):	
		super().__init__(env)
	
	def observation(self, obs):
		# Normalize observation
		return obs/255

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # Clip reward between 0 to 1
        return np.clip(reward, 0, 1)
    
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        if action == 3:
            return random.choice([0,1,2])
        else:
            return action


env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
wrapped_env = ObservationWrapper(RewardWrapper(ActionWrapper(env)))

obs = wrapped_env.reset()

for step in range(500):
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    
    # Raise a flag if values have not been vectorised properly
    if (obs > 1.0).any() or (obs < 0.0).any():
        print("Max and min value of observations out of range")
    
    # Raise a flag if reward has not been clipped.
    if reward < 0.0 or reward > 1.0:
        assert False, "Reward out of bounds"
    
    # Check the rendering if the slider moves to the left.
    #wrapped_env.render()
    
    time.sleep(0.001)

wrapped_env.close()

print("All checks passed")



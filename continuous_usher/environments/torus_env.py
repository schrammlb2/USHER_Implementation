import gym
from gym import spaces
import numpy as np
import pdb

class Torus(gym.GoalEnv):
	def __init__(self, dimension, freeze=False):

		self.state_dim = dimension
		self.goal_dim = dimension
		self.action_dim = dimension + freeze
		self.freeze = freeze

		self.obs_low = np.array([0] * self.state_dim)
		self.obs_high = np.array([1] * self.state_dim)
		self.observation_space = spaces.Dict(dict(
		    desired_goal	=spaces.Box(0, 1, shape= (self.goal_dim,), dtype='float32'),
		    achieved_goal	=spaces.Box(0, 1, shape= (self.goal_dim,), dtype='float32'),
		    observation 	=spaces.Box(0, 1, shape= (self.state_dim + 1,), dtype='float32'),
		))

		self.action_space = spaces.Box(-1, 1, shape= (self.action_dim,), dtype='float32')
		self.dt = .05
		self.min_reward = -1
		self.max_reward = 0
		self.threshold = .2
		# pdb.set_trace()

	def step(self, action):
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/(l1_norm + .0001)
		if self.freeze: 
			# freeze_chance = (action[-1] + 1)/2
			# freeze_chance = action[-1]**2
			freeze_chance = action[-1]
			# freeze_chance = 0 if action[-1] < 0 else 1
			move_action = action[:-1]
		else: 
			freeze_chance = 0
			move_action = action

		if not self.frozen:
			if self.freeze and np.random.rand() < freeze_chance:
				self.freeze_state()
			self.state = self.state + move_action*self.dt

		obs = self.get_obs()
		reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
		assert reward <= self.max_reward and reward >= self.min_reward
		return obs, reward, False, {"is_success": reward == self.max_reward}

	def freeze_state(self):
		self.state = self.observation_space['desired_goal'].sample()
		self.frozen = True

	def reset(self): 
		self.state = self.observation_space['desired_goal'].sample()
		self.goal = self.observation_space['desired_goal'].sample()
		self.frozen = False
		return self.get_obs()

	def get_obs(self): 
		return {'observation': np.concatenate([self.state, np.array([self.frozen*1])]),
				'achieved_goal': self.state, 
				'desired_goal': self.goal}

	def compute_reward(self, a, b, info=None):
		return (((a-b)**2).sum(axis=-1)**.5 < self.threshold) - 1

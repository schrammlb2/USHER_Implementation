
import gym 						#type: ignore
from gym.core import GoalEnv	#type: ignore
from gym import error			#type: ignore
from gym.spaces import Box		#type: ignore
from gym import spaces


import numpy as np 				#type: ignore
from numpy.linalg import norm
import random
import typing
import pdb
# import constants
from constants import *
# from obstacles
import math

from math_utils import rotate


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

ADD_ZERO = True

DISPLAY = True
DISPLAY = False
STEPS = 5

force_long_road = False
force_short_road = False
assert not (force_long_road and force_short_road)
if force_long_road:
	BLOCK_ALT_PATH = False
	SUCCESS_CHANCE = 0
else: 
	if force_short_road:
		BLOCK_ALT_PATH = True
	else: 
		BLOCK_ALT_PATH = False
	# SUCCESS_CHANCE = .5
	SUCCESS_CHANCE = .25
	# SUCCESS_CHANCE = .15
	# SUCCESS_CHANCE = .1
NONBREAKING_FAILURE_CHANCE = .6
HIGH_FAILURE_CHANCE = 1000.#.3
LOW_FAILURE_CHANCE = .9#HIGH_FAILURE_CHANCE/3

break_chance = 10000.0#.2#.6
# BREAKING = False
BREAKING = True
obstacle_density = 0.1
GRID_DISPLAY = False#True
DT = .75
observation_bias = 1.0
base_size = 8

transitions = { 
	EMPTY: lambda last_state, state, dt=1: (state, False),			#Just move
	# BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	# BLOCK: lambda last_state, state: (last_state, True),	#Prevent agent from moving
	BLOCK: lambda last_state, state, dt=1: (last_state, True if random.random() < break_chance*dt else False),	#Prevent agent from moving
	WIND:  lambda last_state, state, dt=1: (state + state_noise(4), False),
	BREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
		else (last_state, BREAKING),
	LOWCHANCE_BREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > LOW_FAILURE_CHANCE*dt \
		else (last_state, BREAKING),
	NONBREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
		else (last_state, False),
}


stopping = { 
	EMPTY: lambda dt=1: (False, False),			#Just move
	# BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	# BLOCK: lambda last_state, state: (last_state, True),	#Prevent agent from moving
	# BLOCK: lambda dt=1: (True, True) if random.random() < break_chance*dt else (True, False),	#Prevent agent from moving
	# BREAKING_DOOR: lambda dt=1: (False, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
	# 	else (True, BREAKING),
	BLOCK: lambda dt=1: (True, True),
	BREAKING_DOOR: lambda dt=1: (True, True),
	LOWCHANCE_BREAKING_DOOR: lambda dt=1: (False, False) if random.random() > LOW_FAILURE_CHANCE*dt \
		else (True, BREAKING),
	NONBREAKING_DOOR: lambda dt=1: (False, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
		else (True, False),
}

# is_unblocked = { 
# 	EMPTY: lambda : True,			
# 	BLOCK: lambda : False,
# 	WIND:  lambda : True,
# 	RANDOM_DOOR: lambda : True if random.random() < SUCCESS_CHANCE else False
# }

def state_noise(k):
	return random.sample(noise_samples + [(0,0)]*k, 1)

def state_normalize(s, size):
	return s*2/size - 1

def state_denormalize(s, size):
	return (s+1)*size/2

def rotate(s, theta):
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	return rot@s

rand_displacement = lambda c : (1-c)/2 + c*np.random.rand(2)



def generate_random_map(size):
	if DISPLAY: print("Generating random map")
	size = size
	mid = size//2
	offset = 2
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	grid = np.zeros((size, size))

	for i in range(size):
		block_chance = obstacle_density
		for j in range(size):
			# if np.random.rand() < block_chance:
			# 	# grid[i, j] = BLOCK
			# 	grid[i, j] = LOWCHANCE_BREAKING_DOOR
			if np.random.rand() < block_chance:
				# grid[i, j] = BREAKING_DOOR
				grid[i, j] = BLOCK
				# grid[i, j] = LOWCHANCE_BREAKING_DOOR
			if np.random.rand() < block_chance:
				# grid[i, j] = LOWCHANCE_BREAKING_DOOR
				grid[i, j] = BLOCK

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

	adj = [-1, 0, 1]
	if DISPLAY: print(grid)
	return grid



def generate_blocky_random_map(size):
	if DISPLAY: print("Generating blocky random map")
	size = size
	mid = size//2
	offset = 2
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	grid = np.zeros((size, size))

	mean_length = 2
	num_squares = (size-1)**2
	block_fraction = obstacle_density

	def assign_blocks(block_type, block_fraction=block_fraction):
		for _ in range(int(num_squares*block_fraction/mean_length)):
			loc = np.random.randint(offset, size-offset, size=2).squeeze()
			# pdb.set_trace()
			grid[tuple(loc)] = block_type
			while not np.random.rand() < 1/(mean_length+1):
				step_loc = lambda loc : (loc + np.array(random.sample(noise_samples, 1)).squeeze())%size
				# step_loc = lambda loc : (loc + np.array(random.sample(noise_samples, 1)))%size
				# step_loc = lambda loc: random.sample([(loc[0]+1, loc[1]),(loc[0]-1, loc[1]),(loc[0], loc[1]+1),(loc[0], loc[1]),-1], 1)
				loc = step_loc(loc)
				grid[tuple(loc)] = block_type
	# assign_blocks(LOWCHANCE_BREAKING_DOOR)
	# assign_blocks(NONBREAKING_DOOR, block_fraction)
	assign_blocks(BREAKING_DOOR, block_fraction)
	assign_blocks(LOWCHANCE_BREAKING_DOOR)
	# assign_blocks(BLOCK, block_fraction=block_fraction/2)

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK
	if DISPLAY: print(grid)

	return grid


def is_solvable(grid):
	moves = [(0,1), (0,-1), (1,0), (-1,0)]
	moves = [np.array(m) for m in moves]
	marked_grid = grid.copy()
	size  = grid.shape[0]
	is_valid = True
	abort = False
	for i in range(size):
		for j in range(size):
			if marked_grid[i,j] == EMPTY and not abort:
				marked_grid[i,j] = -1
				abort = True
	
	if (marked_grid > -1).all(): return False #Entire grid blocked

	loc = lambda i, j, move: tuple(np.array([i,j]) + move)

	progress_made = True
	#Check if every empty tile is reachable from every other one
	#Start with a random empty tile, mark all adjacent ones as reachable
	while progress_made:
		progress_made = False
		for i in range(size):
			for j in range(size):
				if marked_grid[i,j] == -1: 
					for move in moves:
						new_loc = loc(i, j, move)
						if marked_grid[new_loc] == EMPTY:
							marked_grid[new_loc] = -1
							progress_made = True

	is_valid = not (marked_grid == EMPTY).any()

	return is_valid


def two_door_environment(block_start=False):
	size = 5
	mid = 2
	start  = np.array([mid,mid -1])
	new_goal  = np.array([mid, mid +1])
	gridworld = OldGridworldEnv(size, start, new_goal)
	if block_start:
		gridworld.grid[tuple(start)] = BLOCK
	for i in range(size):
		#Borders
		gridworld.grid[0,i] = BLOCK
		gridworld.grid[size-1,i] = BLOCK
		gridworld.grid[i,0] = BLOCK
		gridworld.grid[i, size-1] = BLOCK

		#Wall through the middle
		gridworld.grid[i,mid] = BLOCK


	gridworld.grid[1,mid] = BREAKING_DOOR
	gridworld.grid[-2,mid] = NONBREAKING_DOOR
	return gridworld
	

def get_dynamics(env_type, size):
	if env_type == "linear":
		env =  SimpleDynamicsEnv(size)
	elif env_type == "asteroids": 
		env = AsteroidsDynamicsEnv(size)
	elif env_type == "standard_car": 
		env = StandardCarDynamicsEnv(size)
	elif env_type == "car": 
		env = CarDynamicsEnv(size)
	else: 
		print(f"No dynamics environment matches name {env_type}")
		raise Exception
	return env


def raytrace(start, vector, grid):
	k=20
	distance = .1
	increment = 1.2
	for i in range(k):
		pt = np.clip(start + (distance*increment**i)*vector, 0.01, grid.shape[0]-.01)
		if grid[tuple(pt.astype(int))] != EMPTY: 
			return 1-i/k
	return 0

def get_raytraces(start, grid, num_beams=16):
	angles  = [i/num_beams*2*math.pi for i in range(num_beams)]
	vectors = [np.array([math.cos(theta), math.sin(theta)]) for theta in angles] 
	feedback= [raytrace(start, vector, grid) for vector in vectors]
	return feedback

class SimpleDynamicsEnv:#(gym.GoalEnv):
	def __init__(self, size):
		self.action_dim = 2
		self.state_dim = 2
		self.goal_dim = 2

		self.size = size
		self.obs_low = np.array([0, 0])
		self.obs_high = np.array([self.size - 1, self.size-1])
		self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype='float32')

		self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

	def dynamics(self, state, action, dt):
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/(l1_norm + .0001)
		return (state + action*dt)
		# pos =  (state['pos'] + action*dt)
		# return {"pos": pos, "rot": 0}

	def reset(self, initial_position): 
		return initial_position
		# return {"pos": initial_position, "rot": 0}

	def state_to_obs(self, state) -> np.ndarray:
		return state_normalize(state, self.size)
		# return state['pos']

	def state_to_goal(self, state) -> np.ndarray:
		return state
		# return state['pos']

	def state_to_rot(self, state) -> np.ndarray: 
		return 0

	def stop(self, proposed_invalid_state, prev_state): return prev_state



class AsteroidsDynamicsEnv:
	def __init__(self, size):
		self.action_dim = 2
		self.state_dim = 5
		self.goal_dim = 2

		self.acc_speed = 5
		self.rot_speed = 2

		self.size = size
		self.translation_speed= 1#1##self.size/2
		self.obs_low = np.array([0, 0, -1, -1, -1])
		self.obs_high = np.array([self.size - 1, self.size-1, 1, 1, 1])
		self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype='float32')
		self.action_space = Box(np.array([-1,-1,-1]), np.array([1,1,1]))

	def dynamics(self, state, action, dt):
		action = np.clip(action, -1, 1)

		new_rotation = (state['rot'] + action[1]*dt*self.rot_speed)%(2*math.pi)
		offset = .1
		gas = (action[0] + offset)/(1+offset)
		new_acceleration = np.array([gas*math.cos(new_rotation), gas*math.sin(new_rotation)])
		acc_level = np.clip(self.acc_speed*dt, 0, 1)
		new_velocity = state['vel']*(1-acc_level) + new_acceleration*acc_level
		norm = np.linalg.norm(new_velocity, ord=2)
		new_velocity = new_velocity if norm <= 1 else new_velocity/(norm + .0001)
		new_position = state['pos'] + new_velocity*dt*self.translation_speed
		assert ((new_position - state['pos'])**2).sum()**.5 <= self.translation_speed*1.0001

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state


	def reset(self, initial_position):
		state= {'pos': initial_position, 
				'vel': np.zeros(2),
				'rot': math.pi/2}
		return state

	def state_to_obs(self, state) -> np.ndarray:
		return np.concatenate([state_normalize(state['pos'], self.size), state['vel'], np.array([math.cos(state['rot']), math.sin(state['rot'])])])

	def state_to_goal(self, state) -> np.ndarray:
		return state['pos']

	def state_to_rot(self, state) -> np.ndarray: 
		return state['rot']

	def stop(self, proposed_invalid_state, prev_state):
		new_state ={'pos': prev_state['pos'],
					'vel': prev_state['pos'], 
					'rot': proposed_invalid_state['rot']}
		return new_state

class CarDynamicsEnv(AsteroidsDynamicsEnv):
	def __init__(self, size):
		super().__init__(size)
		self.rot_speed=10
		self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

	def dynamics(self, state, action, dt):
		turn = action[1]
		heading = np.array([math.cos(state['rot']), math.sin(state['rot'])])
		new_rotation = (state['rot'] + norm(state['vel'])*turn*dt*self.rot_speed)%(2*math.pi)
		gas = (action[0] + .5)*2/3
		new_acceleration = gas*heading
		new_velocity = new_acceleration*heading
		vel_norm = norm(new_velocity, ord=2)
		new_velocity = new_velocity if vel_norm <= 1 else new_velocity/(vel_norm + .0001)
		new_position = state['pos'] + new_velocity*dt*self.translation_speed
		assert ((new_position - state['pos'])**2).sum()**.5 <= self.translation_speed*1.00001

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state


class StandardCarDynamicsEnv(AsteroidsDynamicsEnv):
	def __init__(self, size):
		self.action_dim = 2
		self.state_dim = 5
		self.goal_dim = 2

		self.acc_speed = 2
		self.rot_speed = 2

		self.size = size
		self.translation_speed= 1#1##self.size/2
		self.obs_low = np.array([0, 0, -1, -1, -1])
		self.obs_high = np.array([self.size - 1, self.size-1, 1, 1, 1])
		self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype='float32')
		# self.action_space = Box(np.array([-1,-1,-1]), np.array([1,1,1]))
		self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

		self.length = 1/self.rot_speed
		self.wheel_max_turn = 80*(math.pi/180)#1

	def dynamics(self, state, action, dt):
		offset = .1
		gas = (action[0] + offset)/(1+offset)
		# gas = action[0]
		turn = action[1]

		new_speed = gas
		heading = np.array([math.cos(state['rot']), math.sin(state['rot'])])
		new_velocity = new_speed*heading

		rotation_speed = new_speed/self.length*math.tan(turn*self.wheel_max_turn)
		new_rotation = (state['rot'] + rotation_speed*dt)%(2*math.pi)
		vel_norm = norm(new_velocity, ord=2)
		new_velocity = new_velocity if vel_norm <= 1 else new_velocity/(vel_norm + .0001)
		new_position = state['pos'] + new_velocity*dt*self.translation_speed
		assert ((new_position - state['pos'])**2).sum()**.5 <= self.translation_speed*1.00001

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state

	def reset(self, initial_position):
		state= {'pos': initial_position, 
				'vel': np.zeros(2),
				'rot': math.pi/2}
		return state

class AltGridworldEnv(GoalEnv):
	def __init__(self, size, start, new_goal, grid, dynamics, randomize_start=False):
		self.dim = 2
		self.size = size
		self.start = start
		self.new_goal = new_goal
		self.grid = grid
		self.env = dynamics

		if ADD_ZERO: 
			self.obs_scope = (size, size, 2)
		else:
			self.obs_scope = (size, size)

		self.goal_scope = (size, size)

		self.min_reward = -1
		self.max_reward = 0
		self.reward_range = (self.min_reward, self.max_reward)
		self.steps = STEPS
		self.high_speed_pretraining = True
		self.pretrain_iters = 0#250
		self.base_dt = DT#.25
		x = .6
		self.width = x
		self.length = x


		self.randomize_start = randomize_start
		self.size = self.grid.shape[0] - 1
		self.observation_space = spaces.Dict(dict(
		    desired_goal	=spaces.Box(0, self.size, shape= (2,), dtype='float32'),
		    achieved_goal	=spaces.Box(0, self.size, shape= (2,), dtype='float32'),
		    observation 	=self.env.observation_space,
		))
		self.action_space = self.env.action_space
		self.grid_resetter = lambda : self.grid

		self.visualize = False


	def set_grid_resetter(self, grid_resetter):
		self.grid_resetter = grid_resetter
		self.solvable_reset()
		self.safe_maps = []
		for _ in range(100):
			self.solvable_reset()
			if GRID_DISPLAY: print(self.grid)
			self.safe_maps.append(self.grid.copy())
		self.select_safe_map()

		
	def solvable_reset(self):
		self.grid = self.grid_resetter()
		while not is_solvable(self.grid):
			self.grid = self.grid_resetter()

	def select_safe_map(self):
		index = np.random.randint(len(self.safe_maps))
		self.grid = self.safe_maps[index]
		self.one_hot = np.zeros(len(self.safe_maps))
		self.one_hot[index] = 1

	def reset(self):
		c = 0
		self.select_safe_map()
		def random_offset(c): 
			return .5 + np.random.uniform(low=np.array([-c,-c]), high=np.array([c,c]))

		self.broken = False
		if self.randomize_start or np.random.rand() < .01:
			state = self.observation_space['desired_goal'].sample()
			while self.grid[tuple(state.astype(int))] != EMPTY:
				state = self.observation_space['desired_goal'].sample()
			self.goal = self.observation_space['desired_goal'].sample()
			while self.grid[tuple(self.goal.astype(int))] != EMPTY:
				self.goal = self.observation_space['desired_goal'].sample()
		else: 
			state = self.start + random_offset(0.0) 
			self.goal = self.new_goal + random_offset(1.)#+ rand_displacement(1)

		self.state = self.env.reset(state)
		self.path = []
		self.path.append(self.env.state_to_goal(self.state))
		return self.get_obs()

	def check_collisions(self, proposed_next_state, dt):
		proposed_ag = self.env.state_to_goal(proposed_next_state)
		corner_offsets = []
		for x in [-self.width/2, self.width/2]:
			for y in [-self.length/2, self.length/2]:
				corner_offsets.append(np.array([x, y]))
		positions = [proposed_ag] + [proposed_ag + corner_offset for corner_offset in corner_offsets]
		positions = [np.clip(p, .001, self.grid.shape[0] - .001) for p in positions]
		stopped, broken = False, False
		for pos in positions:
			next_state_type = self.grid[tuple(pos.astype(int))]
			next_stopped, next_broken = stopping[next_state_type](dt/len(positions))
			stopped = stopped or next_stopped
			broken = broken or next_broken
		return stopped, broken

	def state_to_obs(self, state) -> np.ndarray:
		return self.env.state_to_obs(state)

	def state_to_goal(self, state) -> np.ndarray:
		return self.env.state_to_goal(state)

	def state_to_rot(self, state) -> np.ndarray: 
		return self.env.state_to_rot(state)

	def step(self, action):
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/l1_norm
		began_broken = self.broken
		state = self.state
		last_state = self.state.copy()
		proposed_next_state = state
		next_state = state
		broken = self.broken
		dt = 1/self.steps*self.base_dt
		for _ in range(self.steps):
			proposed_next_state = self.env.dynamics(next_state, action, dt)
			stopped, next_broken = self.check_collisions(proposed_next_state, dt)
			if not stopped and not broken: 
				next_state = proposed_next_state
			broken = broken or next_broken
			self.path.append(self.env.state_to_goal(next_state))
			if broken: break

		if broken: 
			self.broken = True
		if self.broken:
			next_state = state.copy()

		self.state = next_state.copy()
		reward = self.compute_reward(self.env.state_to_goal(next_state), self.goal)
		try: 
			assert type(reward) == int or type(reward) == np.int64
			assert (reward <= self.max_reward).all() and (reward >= self.min_reward).all()
		except: 
			pdb.set_trace()
		observation = self.get_obs()
		return observation, reward, False, {"is_success": reward == self.max_reward}


	def same_square(self, ag, dg, info=None):
		in_same_square = (ag.astype(int) == dg.astype(int)).all(axis=-1) + 0
		return in_same_square

	def nearby(self, ag, dg):
		threshold = 1
		return (((ag - dg)**2).sum(axis=-1) < threshold) + 0

	def compute_reward(self, ag, dg, info=None):
		is_nearby = self.nearby(ag, dg)
		return is_nearby*self.max_reward + (1-is_nearby)*self.min_reward

	def rand_state(self):
		return np.array([np.random.randint(0, size), np.random.randint(0, size)])

	def set_state(self, state): 
		self.state = state


	def get_state_obs(self): 
		rv = np.append(self.env.state_to_obs(self.state), self.broken)		
		return rv

	def get_goal(self): 
		return self.env.state_to_goal(self.state)

	def type_to_failchance(self, blocktype):
		if blocktype == BLOCK: 
			chance = 1
		elif blocktype == BREAKING_DOOR:
			chance = HIGH_FAILURE_CHANCE
		elif blocktype == LOWCHANCE_BREAKING_DOOR:
			chance = LOW_FAILURE_CHANCE
		elif blocktype == NONBREAKING_DOOR:
			chance = NONBREAKING_FAILURE_CHANCE
		elif blocktype == EMPTY:
			chance = 0
		return chance

	def get_raytraces(self, ag, grid):
		return get_raytraces(ag, grid)

	def get_obs(self):
		vals = [-1, 0, 1]
		moves = [np.array([i,j]) for i in vals for j in vals if (i,j) != (0,0)]
		moves_list = []
		for i in range(3):
			moves_list += [move*i for move in moves]
		moves = moves_list


		state_obs = self.get_state_obs()
		ag = self.env.state_to_goal(self.state)

		surroundings = []
		surroundings += self.get_raytraces(ag, self.grid)
		map_grid = self.grid[1:-1, 1:-1].flatten()
		ttf = np.vectorize(self.type_to_failchance)
		grid_obs = ttf(map_grid)
		return {
			"state": state_obs,
			"observation": np.concatenate([state_obs, surroundings]),
			# "observation": np.concatenate([state_obs]),#, self.one_hot]),
			"achieved_goal": self.get_goal(),
			"desired_goal": self.goal
		}


class CalibrationErrorEnv(AltGridworldEnv):
	def reset(self):
		self.bias_scale =observation_bias
		self.state_bias = 0
		self.rangefinder_bias = 0
		super().reset()
		obs = self.env.state_to_obs(self.state)
		self.state_bias = np.random.normal(0, self.bias_scale, size=obs.shape)
		self.state_bias[2:] *= 0
		self.rangefinder_bias = np.random.normal(0, self.bias_scale)
		return self.get_obs()

	def get_state_obs(self): 
		rv = np.append(self.env.state_to_obs(self.state) + self.state_bias, self.broken)		
		return rv

	def get_raytraces(self, ag, grid):
		return [r + self.rangefinder_bias for r in get_raytraces(ag, grid)]


def create_map_1_grid(size, block_start=False):
	grid = create_empty_map_grid(size)
	mid = size//2
	#Wall through the middle
	for i in range(1, size//2 + 1):
		grid[i,mid ] = BLOCK
	grid[2,mid ] = EMPTY
	if DISPLAY: print(grid)
	return grid

def create_map_2_grid(size, block_start=False):
	grid = create_empty_map_grid(size)
	mid = size//2
	#Wall through the middle
	for i in range(1, size - 1):
		grid[i,mid ] = BLOCK
	grid[2,mid ] = EMPTY
	if DISPLAY: print(grid)
	return grid

def create_map_3_grid(size, block_start=False):
	grid = create_empty_map_grid(size)
	mid = size//2
	#Wall through the middle
	for i in range(1, size//2 + 1):
		grid[i,mid ] = BLOCK
	if DISPLAY: print(grid)
	return grid

def create_test_map_grid(size, block_start=False):
	grid = np.zeros((size, size))
	mid = size//2
	if block_start:
		grid[tuple(start)] = BLOCK
		grid[tuple(start + np.array([1, 1]))] = BLOCK

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

		#Wall through the middle
	grid[1,mid] 	= BREAKING_DOOR
	grid[2,mid ] 	= LOWCHANCE_BREAKING_DOOR
	
	if not BLOCK_ALT_PATH:
		grid[size-3,mid] = EMPTY

	if DISPLAY: print(grid)
	return grid



def create_empty_map_grid(size):
	grid = np.zeros((size, size))
	mid = size//2
	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

	if DISPLAY: print(grid)
	return grid



def random_map(env_type="linear"): 
	size = base_size
	grid = generate_random_map(size)
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	randomize_start = True
	dyn = get_dynamics(env_type, size)
	env = AltGridworldEnv(size, start, goal, grid, dyn, randomize_start=randomize_start)
	env.set_grid_resetter(lambda : generate_random_map(size))
	return env

def random_blocky_map(env_type="linear"): 
	size = base_size
	grid = generate_blocky_random_map(size)
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	randomize_start = True
	dyn = get_dynamics(env_type, size)
	env = AltGridworldEnv(size, start, goal, grid, dyn, randomize_start=randomize_start)
	env.set_grid_resetter(lambda : generate_blocky_random_map(size))
	return env


def create_map_1(env_type="linear", block_start=False):
	size = 9
	grid = create_map_1_grid(size, block_start)
	mid = size//2
	offset = np.array([0.5, 0])
	start  = np.array([2,mid -2]) + offset
	goal  = np.array([2, mid +2]) + offset
	randomize_start = False
	dyn = get_dynamics(env_type, size)
	env = AltGridworldEnv(size, start, goal, grid, dyn, randomize_start=randomize_start)
	env.set_grid_resetter(lambda : grid)
	return env




def create_test_map(env_type="linear", block_start=False):
	size = 6
	use_obstacles = True
	if use_obstacles: 
		grid = create_map_1_grid(size, block_start)
		randomize_start = False
	else: 
		grid = create_empty_map_grid(size)
		randomize_start = True
	mid = size//2
	start  = np.array([1,mid -2])
	goal  = np.array([1, mid +1])
	dyn = get_dynamics(env_type, size)
	env = AltGridworldEnv(size, start, goal, grid, dyn, randomize_start=randomize_start)
	env.env = dyn
	env.grid = grid
	return env



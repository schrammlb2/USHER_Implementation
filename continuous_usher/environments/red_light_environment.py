
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


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

ADD_ZERO = True

# DISPLAY = True
DISPLAY = False
STEPS = 5

NONBREAKING_FAILURE_CHANCE = .6
HIGH_FAILURE_CHANCE = 1.0
LOW_FAILURE_CHANCE = .9

break_chance = 0
BREAKING = True
obstacle_density = 0

transitions = { 
	EMPTY: lambda last_state, state, color, dt: (state, False),			#Just move
	BLOCK: lambda last_state, state, color, dt: (last_state, True if random.random() < break_chance*dt else False),	#Prevent agent from moving
	WIND:  lambda last_state, state, color, dt: (state + state_noise(4), False),
	BREAKING_DOOR: lambda last_state, state, color, dt: (state, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
		else (last_state, BREAKING),
	LOWCHANCE_BREAKING_DOOR: lambda last_state, state, color, dt: (state, False) if random.random() > LOW_FAILURE_CHANCE*dt \
		else (last_state, BREAKING),
	NONBREAKING_DOOR: lambda last_state, state, color, dt: (state, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
		else (last_state, False),
	RED_LIGHT_DOOR: lambda last_state, state, color, dt: red_light(last_state, state, color, dt)
}



def red_light(last_state, state, color, dt):
	if color == GREEN or color == YELLOW:
		return transitions[EMPTY](last_state, state, color, dt)
	elif color == RED: 
		return transitions[BREAKING_DOOR](last_state, state, color, dt)
	assert False, f"Invalid light color: {color}"

stopping = { 
	EMPTY: lambda color, dt: (False, False),			#Just move
	# BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	# BLOCK: lambda last_state, state: (last_state, True),	#Prevent agent from moving
	BLOCK: lambda color, dt: (True, True) if random.random() < break_chance*dt else (True, False),	#Prevent agent from moving
	BREAKING_DOOR: lambda color, dt: (False, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
		else (True, BREAKING),
	LOWCHANCE_BREAKING_DOOR: lambda color, dt: (False, False) if random.random() > LOW_FAILURE_CHANCE*dt \
		else (True, BREAKING),
	NONBREAKING_DOOR: lambda color, dt: (False, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
		else (True, False),
	RED_LIGHT_DOOR: lambda color, dt: red_light(color, dt)
}

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



def get_dynamics(env_type, size):
	if env_type == "linear":
		env =  SimpleDynamicsEnv(size)
	elif env_type == "yaxis":
		env =  YAxisDynamicsEnv(size)
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
	distance = .2
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
		return state
		# return state['pos']

	def state_to_goal(self, state) -> np.ndarray:
		return state
		# return state['pos']

	def stop(self, proposed_invalid_state, prev_state): return prev_state



class YAxisDynamicsEnv:#(gym.GoalEnv):
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
		action[0] *= .1
		return (state + action*dt)
		# pos =  (state['pos'] + action*dt)
		# return {"pos": pos, "rot": 0}

	def reset(self, initial_position): 
		return initial_position
		# return {"pos": initial_position, "rot": 0}

	def state_to_obs(self, state) -> np.ndarray:
		return state
		# return state['pos']

	def state_to_goal(self, state) -> np.ndarray:
		return state
		# return state['pos']

	def stop(self, proposed_invalid_state, prev_state): return prev_state


class AsteroidsDynamicsEnv:#(gym.GoalEnv):
	def __init__(self, size):
		self.action_dim = 2
		self.state_dim = 5
		self.goal_dim = 2

		self.acc_speed = 2
		self.rot_speed = 5

		self.size = size
		self.translation_speed= 1#1##self.size/2
		self.obs_low = np.array([0, 0, -1, -1, -1])
		self.obs_high = np.array([self.size - 1, self.size-1, 1, 1, 1])
		self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype='float32')
		self.action_space = Box(np.array([-1,-1,-1]), np.array([1,1,1]))

	def dynamics(self, state, action, dt):
		action = np.clip(action, -1, 1)

		new_rotation = (state['rot'] + action[1]*dt*self.rot_speed)%(2*math.pi)
		# new_rotation = action[1]*(2*math.pi)
		# gas = action[0]/2 + .5
		gas = (action[0] + .5)*2/3
		new_acceleration = np.array([gas*math.cos(new_rotation), gas*math.sin(new_rotation)])
		acc_level = np.clip(self.acc_speed*dt, 0, 1)
		new_velocity = state['vel']*(1-acc_level) + new_acceleration*acc_level
		# new_velocity = new_acceleration

		# new_velocity = action
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
				'vel': np.random.rand(2)*2 - 1, #np.zeros(2),
				'rot': np.random.rand()*2*math.pi}
		# state= {'pos': initial_position, 
		# 		'vel': np.zeros(2),
		# 		'rot': np.random.rand()*2*math.pi}
		return state

	def state_to_obs(self, state) -> np.ndarray:
		# return np.concatenate([state['pos'], state['vel'], np.array([state['rot']])/math.pi - 1])
		return np.concatenate([state_normalize(state['pos'], self.size), state['vel'], np.array([math.cos(state['rot']), math.sin(state['rot'])])])

	def state_to_goal(self, state) -> np.ndarray:
		return state['pos']

	def stop(self, proposed_invalid_state, prev_state):
		new_state ={'pos': prev_state['pos'],
					# 'vel': np.zeros(2), 
					'vel': prev_state['pos'], 
					'rot': proposed_invalid_state['rot']}
					#allow it to turn when it's run up against a wall, rather than just sticking there
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
		# new_velocity = state['vel']*(1-self.acc_speed*dt) + new_acceleration*self.acc_speed*dt
		# new_velocity = new_acceleration
		# new_velocity = (new_velocity@heading)*heading
		new_velocity = new_acceleration*heading
		# new_velocity = np.clip(new_velocity, -1, 1)
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
		self.rot_speed = 5#0.00001

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
		# gas = (action[0] + .5)*2/3
		gas  = action[0]
		action[1] *= .2
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

class TrafficLight: 
	def __init__(self, green_time, yellow_time, red_time):
		self.green_time = green_time
		self.yellow_time = yellow_time
		self.red_time = red_time

		self.total_time = green_time + yellow_time + red_time
		self.current_time = 0

	def get_color(self):
		if self.current_time < self.green_time: 
			current_color = GREEN
		elif self.current_time < self.green_time + self.yellow_time:
			current_color = YELLOW
		else: 
			current_color = RED
		return current_color

	def reset(self):
		self.current_time = np.random.rand()*self.total_time
		return self.get_color()

	def tick(self, dt):
		self.current_time = (self.current_time + dt)%self.total_time
		return self.get_color()



# # Version that the positive results were gathered with
class AltGridworldEnv(GoalEnv):
	def __init__(self, size, start, new_goal, grid, dynamics, randomize_start=False):
		self.dim = 2
		self.size = size
		self.start = start
		self.new_goal = new_goal
		self.grid = grid
		self.env = dynamics
		print(dynamics)

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
		# self.env = SimpleDynamicsEnv(size, start)
		self.pretrain_iters = 0#250
		# self.env.translation_speed = 1#self.size/10
		self.base_dt = 1.0#.25


		self.light = TrafficLight(1, 1, 10)


		self.randomize_start = False
		self.size = self.grid.shape[0] - 1
		self.observation_space = spaces.Dict(dict(
		    desired_goal	=spaces.Box(0, self.size, shape= (2,), dtype='float32'),
		    achieved_goal	=spaces.Box(0, self.size, shape= (2,), dtype='float32'),
		    observation 	=self.env.observation_space,
		))
		self.action_space = self.env.action_space
		self.grid_resetter = lambda : self.grid


	def set_grid_resetter(self, grid_resetter):
		self.grid_resetter = grid_resetter
		

	def reset(self):
		c = 0
		self.light.reset()
		self.grid = self.grid_resetter()
		self.broken = False
		def random_offset(c): 
			return .5 + np.random.uniform(low=np.array([-c,-c]), high=np.array([c,c]))

		state = self.start + rand_displacement(.5)
		self.goal = self.new_goal + rand_displacement(.5)

		self.state = self.env.reset(state)
		return self.get_obs()


	def step(self, action):
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/l1_norm
		began_broken = self.broken
		state = self.state
		last_state = self.state.copy()
		proposed_next_state = state
		next_state = state
		broken = False
		dt = 1/self.steps*self.base_dt
		color = self.light.get_color()
		for _ in range(self.steps):
			proposed_next_state = self.env.dynamics(next_state, action, dt)
			color = self.light.tick(dt)
			proposed_ag = np.clip(self.env.state_to_goal(proposed_next_state), .001, self.grid.shape[0] - .001)
			next_state_type = self.grid[tuple(proposed_ag.astype(int))]
			next_state, next_broken = transitions[next_state_type](state, proposed_next_state, color, dt)
			broken = broken or next_broken
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
		threshold = 2**(-.5)
		return (((ag - dg)**2).sum(axis=-1) < threshold) + 0

	def compute_reward(self, ag, dg, info=None):
		is_nearby = self.same_square(ag,dg)
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

	def get_obs(self):
		vals = [-1, 0, 1]
		moves = [np.array([i,j]) for i in vals for j in vals if (i,j) != (0,0)]

		moves = moves + [move*2 for move in moves]  + [move/2 for move in moves] + [np.zeros(2)]

		state_obs = self.get_state_obs()
		ag = self.env.state_to_goal(self.state)

		surroundings = []
		surroundings += [self.light.get_color()]
		return {
			"state": state_obs,
			"observation": np.concatenate([state_obs, surroundings]),
			"achieved_goal": self.get_goal(),
			"desired_goal": self.goal
		}




def create_map_1_grid(size, block_start=False):
	grid = create_empty_map_grid(size)
	mid = size//2
	if block_start:
		grid[tuple(start)] = BLOCK
		grid[tuple(start + np.array([1, 1]))] = BLOCK
	for i in range(size):
		#Borders
		grid[size-2,i] = BLOCK

		#Wall through the middle
		grid[i,mid ] = BLOCK

	grid[1,mid] = RED_LIGHT_DOOR
	
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



def create_red_light_map(env_type="linear", block_start=False):
	size = 6
	grid = create_map_1_grid(size, block_start)
	mid = size//2
	start  = np.array([1,mid -2])
	goal  = np.array([1, mid +1])
	randomize_start = False
	dyn = get_dynamics(env_type, size)
	env = AltGridworldEnv(size, start, goal, grid, dyn, randomize_start=randomize_start)
	return env


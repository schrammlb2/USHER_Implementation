
import gym                      #type: ignore
# from gym.core import GoalEnv  #type: ignore
from gym import error           #type: ignore
from gym.spaces import Box      #type: ignore
from gym import spaces


import numpy as np              #type: ignore
from numpy.linalg import norm
import random
import typing
import pdb
# import constants
from constants import *
# from obstacles
import math

from math_utils import rotate
from math import pi,cos,sin


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
HIGH_FAILURE_CHANCE = 0.9#.3
LOW_FAILURE_CHANCE = .9#HIGH_FAILURE_CHANCE/3

break_chance = 0#10000.0#.2#.6
# BREAKING = False
BREAKING = True
obstacle_density = 0.1
GRID_DISPLAY = False#True
DT = .5
observation_bias = 1.0
base_size = 7

BLOCK_TYPE_ENVIRONMENT = True
# UNBLOCKED_CHANCE = 0.4#0.5#0.25#0.6#0.4
UNBLOCKED_CHANCE = 0.4 #Should work with enough exploration
RANDOM_GOAL_CHANCE = 0.2

transitions = { 
    EMPTY: lambda last_state, state, dt=1: (state, False),          #Just move
    # BLOCK: lambda last_state, state: (last_state, False), #Prevent agent from moving
    # BLOCK: lambda last_state, state: (last_state, True),  #Prevent agent from moving
    BLOCK: lambda last_state, state, dt=1: (last_state, True if random.random() < break_chance*dt else False),  #Prevent agent from moving
    WIND:  lambda last_state, state, dt=1: (state + state_noise(4), False),
    BREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
        else (last_state, BREAKING),
    LOWCHANCE_BREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > LOW_FAILURE_CHANCE*dt \
        else (last_state, BREAKING),
    NONBREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
        else (last_state, False),
}


stopping = { 
    EMPTY: lambda dt=1: (False, False),         #Just move
    # BLOCK: lambda last_state, state: (last_state, False), #Prevent agent from moving
    # BLOCK: lambda last_state, state: (last_state, True),  #Prevent agent from moving
    # BLOCK: lambda dt=1: (True, True) if random.random() < break_chance*dt else (True, False), #Prevent agent from moving
    # BREAKING_DOOR: lambda dt=1: (False, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
    #   else (True, BREAKING),
    BLOCK: lambda dt=1: (True, True),
    BREAKING_DOOR: lambda dt=1: (True, True),
    LOWCHANCE_BREAKING_DOOR: lambda dt=1: (False, False) if random.random() > LOW_FAILURE_CHANCE*dt \
        else (True, BREAKING),
    NONBREAKING_DOOR: lambda dt=1: (False, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
        else (True, False),
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
    elif env_type == "asteroids": 
        env = AsteroidsDynamicsEnv(size)
    elif env_type == "standard_car": 
        env = StandardCarDynamicsEnv(size)
    elif env_type == "car": 
        env = CarDynamicsEnv(size)
    elif env_type == "omnibot": 
        env = OmnibotDynamicsEnv(size)
    else: 
        print(f"No dynamics environment matches name {env_type}")
        raise Exception
    return env


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


class AsteroidsDynamicsEnv:#(gym.GoalEnv):
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
        # new_rotation = action[1]*(2*math.pi)
        # gas = action[0]/2 + .5
        offset = .1
        gas = (action[0] + offset)/(1+offset)
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

    # def reset(self, initial_position):
    #   state= {'pos': initial_position, 
    #           'vel': np.random.rand(2)*2 - 1, #np.zeros(2),
    #           'rot': np.random.rand()*2*math.pi}
    #   # state= {'pos': initial_position, 
    #   #       'vel': np.zeros(2),
    #   #       'rot': np.random.rand()*2*math.pi}
    #   return state

    def reset(self, initial_position):
        state= {'pos': initial_position, 
                'vel': np.zeros(2),
                'rot': math.pi/2}
        return state

    def state_to_obs(self, state) -> np.ndarray:
        # return np.concatenate([state['pos'], state['vel'], np.array([state['rot']])/math.pi - 1])
        return np.concatenate([state_normalize(state['pos'], self.size), state['vel'], np.array([math.cos(state['rot']), math.sin(state['rot'])])])

    def state_to_goal(self, state) -> np.ndarray:
        return state['pos']

    def state_to_rot(self, state) -> np.ndarray: 
        return state['rot']

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
        rotation_speed = new_speed/self.length*math.tan(turn*self.wheel_max_turn)
        new_rotation = (state['rot'] + rotation_speed*dt)%(2*math.pi)
        heading = np.array([math.cos(new_rotation), math.sin(new_rotation)])
        new_velocity = new_speed*heading

        rotation_speed = new_speed/self.length*math.tan(turn*self.wheel_max_turn)

        # wheel_turn_limitation = self.wheel_max_turn*(1-new_speed/2)
        # rotation_speed = new_speed/self.length*math.tan(turn*wheel_turn_limitation)

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
        # state= {'pos': initial_position, 
        #       'vel': np.zeros(2),
        #       'rot': np.random.rand()*2*math.pi}
        return state


class OmnibotDynamicsEnv:#(gym.GoalEnv):
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
        self.action_space = Box(np.array([-1,-1,-1,-1]), np.array([1,1,1,1]))

        self.control_torque = 34.5575
        self.mass = 10.
        self.gravity = 9.8
        self.radius = .3
        self.stall_torque = 25.
        self.la = .11
        self.lb = .1
        self.width = .152
        self.height = .23
        self.steps = 15

        self.scale = 1./self.radius
        self.mu = np.array([0.53602521, 0.59749176, 0.80504539, 1.03905589])

        self.A = np.matrix([[-1.,1.,self.la+self.lb],[1.,1.,-(self.la+self.lb)],
                [-1.,1.,-(self.la+self.lb)],[1.,1.,(self.la+self.lb)]]) * self.scale

    def compute_omega(self, mu, action):
        return self.control_torque*action*(1. - (mu * self.mass * self.gravity * self.radius)/(4. * self.stall_torque))

    def state_to_obs(self, state) -> np.ndarray:
        # return np.concatenate([state['pos'], state['vel'], np.array([state['rot']])/math.pi - 1])
        return np.concatenate([state_normalize(state['pos'], self.size), np.array([math.cos(state['rot']), math.sin(state['rot'])])])

    def state_to_goal(self, state) -> np.ndarray:
        return state['pos']

    def state_to_rot(self, state) -> np.ndarray: 
        return state['rot']

    def dynamics(self, state, action, dt):
        action = np.clip(action, -1, 1)
        pos = state['pos']
        #Where does the action come in? 
        omega = np.array([self.compute_omega(self.mu[i], action[i]) for i in range(4)])
        x = np.linalg.pinv(self.A).dot(omega)
        x = np.squeeze(np.asarray(x))

        B = np.matrix([[cos(state['rot']),-sin(state['rot']),0.],[sin(state['rot']),cos(state['rot']),0.],[0.,0.,1.]])  
        vel = np.squeeze(np.asarray(B.dot(x)))

        new_pos = pos + vel[:2]*dt
        new_rot = state['rot'] + vel[2]*dt
        new_state= {
                'pos': new_pos, 
                'rot': new_rot
            }
        return new_state

    def reset(self, initial_position):
        state= {'pos': initial_position, 
                'rot': math.pi/2}
        return state

class Omnibot2DDynamicsEnv(OmnibotDynamicsEnv):
    def __init__(self, size):
        super().__init__(size)
        self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

    def dynamics(self, state, action, dt):
        action_4D = action[0]*np.ones(4) + action[1]*arr([1,1,-1,-1])
        return super().dynamics(state, action, dt)

class AltGridworldEnv(gym.core.Env):
    def __init__(self, size, start, new_goal, grid, dynamics, randomize_start=False):
        self.dim = 2
        self.size = size
        self.start = start
        self.new_goal = new_goal
        self.base_grid = grid
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
        # self.env = SimpleDynamicsEnv(size, start)
        self.pretrain_iters = 0#250
        # self.env.translation_speed = 1#self.size/10
        self.base_dt = DT#.25
        # self.width = 1.0
        # self.length = 1.0#1
        try: 
            self.width = dynamics.width
            self.length = dynamics.height
        except: 
            x = .7
            self.width = x*2/3
            self.length = x


        self.randomize_start = randomize_start
        self.size = self.base_grid.shape[0] - 1
        self.observation_space = spaces.Dict(dict(
            desired_goal    =spaces.Box(0, self.size, shape= (2,), dtype='float32'),
            achieved_goal   =spaces.Box(0, self.size, shape= (2,), dtype='float32'),
            observation     =self.env.observation_space,
        ))
        self.action_space = self.env.action_space

        self.visualize = False



    def reset_map(self):
        self.grid = self.base_grid.copy()
        mid = self.grid.shape[0]//2
        if np.random.rand() < UNBLOCKED_CHANCE:
            if BLOCK_TYPE_ENVIRONMENT: self.grid[2,mid] = 0
            self.block_position = "unblocked"
        else: 
            if BLOCK_TYPE_ENVIRONMENT:  self.grid[2,mid] = 1
            self.block_position = "blocked"


    def reset(self):
        c = 0
        # prev_grid = self.grid.copy()
        self.reset_map()
        # self.solvable_reset()
        # assert (prev_grid == self.grid).all()
        # pdb.set_trace()
        def random_offset(c): 
            return .5 + np.random.uniform(low=np.array([-c,-c]), high=np.array([c,c]))

        self.broken = False
        if np.random.rand() < 1-RANDOM_GOAL_CHANCE:#.95: 
            self.goal = self.new_goal + np.random.uniform(low=np.array([-1,-1]), high=np.array([1,1]))
        else: 
            done = False
            while not done:
                self.goal = np.random.uniform(low=np.zeros(2), high=np.ones(2)*self.grid.shape[0])
                done = (self.grid[tuple(self.goal.astype(int))] < 0.5)

        # pdb.set_trace()
        self.state = self.env.reset(self.start)
        # self.state = self.env.reset(state)
        self.path = []
        self.path.append(self.env.state_to_goal(self.state))
        return self.get_obs()


    def check_collisions(self, proposed_next_state, dt):
        proposed_ag = self.env.state_to_goal(proposed_next_state)
        theta = self.env.state_to_rot(proposed_next_state)
        rot_mat = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
        corner_offsets = []
        for x in [-self.width/2, self.width/2]:
            for y in [-self.length/2, self.length/2]:
                corner_offsets.append(rot_mat(theta)@np.array([x, y]))
        # angle = self.env.state_to_rot(proposed_next_state)
        # positions = [proposed_ag] + [proposed_ag + rotate(corner_offset, angle) for corner_offset in corner_offsets]
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
        # action = action/np.linalg.norm(action)
        l1_norm = np.abs(action).sum()
        if l1_norm > 1: 
            action = action/l1_norm
        # action = np.clip(action, -1, 1)
        began_broken = self.broken
        state = self.state
        last_state = self.state.copy()
        proposed_next_state = state
        next_state = state
        # broken = False
        broken = self.broken
        dt = 1/self.steps*self.base_dt
        for _ in range(self.steps):
            proposed_next_state = self.env.dynamics(next_state, action, dt)
            # proposed_ag = np.clip(self.env.state_to_goal(proposed_next_state), .001, self.grid.shape[0] - .001)
            # next_state_type = self.grid[tuple(proposed_ag.astype(int))]
            # stopped, next_broken = stopping[next_state_type](dt)
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

        # assert (np.abs(state - next_state) < 1.01).all()

        self.state = next_state.copy()
        reward = self.compute_reward(self.env.state_to_goal(next_state), self.goal)
        try: 
            assert type(reward) == int or type(reward) == np.int64
            assert (reward <= self.max_reward).all() and (reward >= self.min_reward).all()
        except: 
            pdb.set_trace()
        observation = self.get_obs()
        # if self.high_speed_pretraining: self.broken = False
        return observation, reward, False, {"is_success": reward == self.max_reward}


    def same_square(self, ag, dg, info=None):
        in_same_square = (ag.astype(int) == dg.astype(int)).all(axis=-1) + 0
        return in_same_square

    def nearby(self, ag, dg):
        threshold = 1#2**(-.5)
        return (((ag - dg)**2).sum(axis=-1) < threshold) + 0

    def compute_reward(self, ag, dg, info=None):
        # if self.randomize_start: 
        #   is_nearby = self.nearby(ag, dg)
        # else: 
        #   is_nearby = self.same_square(ag,dg)
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
        # pdb.set_trace()

    def get_raytraces(self, ag, grid):
        return get_raytraces(ag, grid)

    def get_obs(self):
        state_obs = self.get_state_obs()
        ag = self.env.state_to_goal(self.state)

        return {
            "state": state_obs,
            # "observation": np.concatenate([state_obs, surroundings]),#, self.one_hot]),
            "observation": state_obs,#, self.one_hot]),
            "achieved_goal": self.get_goal(),
            "desired_goal": self.goal
        }




def create_map_1_grid(size, block_start=False):
    grid = create_empty_map_grid(size)
    mid = size//2
    #Wall through the middle
    for i in range(1, size//2 + 1):
    # for i in range(1, size - 1):
        grid[i,mid ] = BLOCK
    # grid[2,mid ] = EMPTY
    grid[2,mid ] = BREAKING_DOOR
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


def create_map_1(env_type="linear", block_start=False, size=base_size):
    print(f"Size: {size}")
    grid = create_map_1_grid(size, block_start)
    mid = size//2
    offset = np.array([0.5, 0])
    start  = np.array([2,mid -1]) + offset
    # start  = np.array([2,mid -1]) + offset
    goal  = np.array([2, mid +2]) + offset
    randomize_start = False
    # randomize_start = True
    # env = get_class_constructor(env_type, size, grid, randomize_start, start, goal)
    # env = OriginalGridworldEnv(size, start, goal)
    # env.grid = grid
    dyn = get_dynamics(env_type, size)
    env = AltGridworldEnv(size, start, goal, grid, dyn, randomize_start=randomize_start)
    return env



if __name__ == '__main__':
    # DISPLAY = True
    env = create_map_1(size=base_size)
    env.reset()
    print(env.grid)
    for i in range(10):
        print(env.step(env.action_space.sample()))

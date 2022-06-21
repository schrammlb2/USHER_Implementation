from __future__ import annotations

import numpy as np
from typing import Tuple, Sequence, Callable, List, Union, TypeVar, Iterable, Optional
from functools import reduce

from alt_gridworld import create_map_1, two_door_environment
# from pure_gridworld import create_map_1
from display import * #type: ignore
from constants import *
import pdb
import matplotlib.pyplot as plt #type: ignore
# from beartype import beartype
from typeguard import typechecked

State=np.ndarray
Goal=np.ndarray
Action=np.ndarray
ActionIndex = Union[int, np.int64]

Transition = Tuple[State, Goal, Goal, ActionIndex, State, float]
# PolicyType = Union[Literal["Q"], Literal["HER"], Literal["USHER"]]
PolicyType = str #Should be enumeration of Q, HER, and USHER, but typing module isnt cooperating

T = TypeVar('T')

def mean(lst):
	return sum(lst)/len(lst)

def softmax(arr: np.ndarray, temp: float) ->  np.ndarray:
	unnormed_vals = 2**(temp*arr)
	return unnormed_vals/unnormed_vals.sum()

def softmax_sample(arr: np.ndarray, temp: float) -> ActionIndex:
	probabilities = softmax(arr, temp)
	return np.random.choice(np.arange(len(arr)), p=probabilities)

def default(x: Optional[T], default_value: T) -> T:
	if x is None: 	
		assert default_value is not None
		return default_value
	else: 			
		assert x is not None
		return x

DOORS = 1

if DOORS == 1:
	env = create_map_1(block_start=True)
	# episodes = int(5*10**3)
	episodes = int(1*10**4)
	base_lr = .01
	gamma = .825
	TEMP = 5
	k = 8
	repeat_num = 4
	traj_steps = 30

elif DOORS == 2: 
	env = two_door_environment(block_start=True)
	episodes = int(2.5*10**4)
	base_lr = .001
	gamma = .7
	TEMP = 5
	k = 16
	repeat_num = 4
	traj_steps = 20

else: 
	print("Invalid map")
	exit()

record_num = 200


USE_HER = True
USE_Q = True
USE_USHER = True
DISPLAY = True
# DISPLAY = False

# @typechecked
class Q: 
	def __init__(self, size: int, compute_reward, default_goal: np.ndarray):
		# self.q_table = np.ones((env.size, env.size, 5))
		shape = env.obs_scope + env.goal_scope + (5,)
		self.q_table = np.zeros(shape) + 1
		self.pure_q_table = np.zeros(shape) + 1
		self.usher_q_table = np.zeros(env.obs_scope + env.goal_scope*2 + (5,)) + 1
		self.probability_table = np.zeros(env.obs_scope + env.goal_scope*2 + (5,)) + 1

		self.q_table[:,:, 1, ...] = 0
		self.pure_q_table[:,:, 1, ...] = 0
		self.usher_q_table[:,:, 1, ...] = 0
		# self.usher_q_table *= 0
		# self.v_table = np.zeros((env.size, env.size))
		self.compute_reward = compute_reward
		self.default_goal = default_goal
		self.p = 1/k

	def sample_action(self, state: np.ndarray, goal: np.ndarray, temp=TEMP, policy: PolicyType = "HER") -> ActionIndex: 
		use_pol_goal = goal 
		if policy == "Q":
			q = self.pure_q_table[tuple(state) + tuple(goal)]
		elif policy == "HER": 
			q = self.q_table[tuple(state) + tuple(goal)]
		elif policy == "USHER":
			q = self.usher_q_table[tuple(state) + tuple(goal) + tuple(use_pol_goal)]

		# pdb.set_trace()
		return softmax_sample(q, temp)

	# def argmax_action(self, state: np.ndarray, goal: np.ndarray) -> ActionIndex:
	# 	q = self.q_table[tuple(state) + tuple(goal)]
	# 	return q.argmax()

	def argmax_action(self, state: np.ndarray, goal: np.ndarray, pol_goal: Optional[np.ndarray]=None, 
		policy: PolicyType = "HER") -> ActionIndex: 
		# use_pol_goal = goal if (type(pol_goal) == None) else pol_goal
		use_pol_goal = default(goal, pol_goal)
		assert use_pol_goal is not None
		if policy == "Q":
			q = self.pure_q_table[tuple(state) + tuple(goal)]
		elif policy == "HER": 
			q = self.q_table[tuple(state) + tuple(goal)]
		elif policy == "USHER":
			q = self.usher_q_table[tuple(state) + tuple(goal) + tuple(use_pol_goal)]
		return q.argmax()

	def state_value(self, state: np.ndarray, policy: PolicyType = "HER") -> float:
		if policy == "Q":
			q = self.pure_q_table[tuple(state) + (0,) + tuple(self.default_goal)]
		elif policy == "HER": 
			q = self.q_table[tuple(state) + (0,) + tuple(self.default_goal)]
		elif policy == "USHER":
			q = self.usher_q_table[tuple(state) + (0,) + tuple(self.default_goal) + tuple(self.default_goal)]
		return q.max()
		# return self.q_table[tuple(state) + (0,) + tuple(self.default_goal)].max()

	def _update_q(self, state: np.ndarray, goal: np.ndarray, achieved_goal: np.ndarray, action: ActionIndex, next_state: np.ndarray, 
			reward: float, lr: float=.05) -> None:
		assert state.shape == next_state.shape
		reward = 1 if (achieved_goal == goal).all() else 0
		bellman_update = (1-gamma)*reward + gamma*self.q_table[tuple(next_state) + tuple(goal)].max()
		a = (1-lr)*self.q_table[tuple(state) + tuple(goal)][action]
		b = lr*bellman_update
		self.q_table[tuple(state) + tuple(goal)][action] = a + b

	def _update_pure_q(self, state: np.ndarray, goal: np.ndarray, achieved_goal: np.ndarray, action: ActionIndex, next_state: np.ndarray, 
			reward: float, lr: float=.05) -> None:
			#Not updated by HER
		assert state.shape == next_state.shape
		reward = 1 if (achieved_goal == goal).all() else 0
		bellman_update = (1-gamma)*reward + gamma*self.pure_q_table[tuple(next_state) + tuple(goal)].max()
		a = (1-lr)*self.pure_q_table[tuple(state) + tuple(goal)][action]
		b = lr*bellman_update
		self.pure_q_table[tuple(state) + tuple(goal)][action] = a + b

	def _update_probability(self, state: np.ndarray, goal: np.ndarray, achieved_goal: np.ndarray, pol_goal: np.ndarray, action: int, 
			next_state: np.ndarray, lr: float=.05, p=None, t_remaining=None) -> None:
		assert state.shape == next_state.shape
		reward = 1 if (achieved_goal == goal).all() else 0
		bellman_update = (1-gamma)*reward + gamma*self.probability_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		a = (1-lr)*self.probability_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = lr*bellman_update
		self.probability_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b

	def _update_off_target_usher_q(self, state: np.ndarray, goal: np.ndarray, achieved_goal: np.ndarray, pol_goal: np.ndarray, action: int, 
			next_state: np.ndarray, lr: float=.05, p=None, t_remaining=None) -> None:
		assert state.shape == next_state.shape

		# ratio_table = self.probability_table
		ratio_table = self.usher_q_table

		reward = 1 if (achieved_goal == goal).all() else 0
		const = .01
		bellman_update = (1-gamma)*reward + gamma*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		if (goal == pol_goal).all():
			p = self.p if p is None else p
			next_action = self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(goal)].argmax()
			ratio = (
				(const + p + (1-p)*ratio_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action])/
				(const + p + (1-p)*ratio_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
				)
		else: 
			p = .01*lr
			ratio = (const + ratio_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]) / \
			(const + ratio_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())	
		new_lr = min(lr*ratio, 1)
		a = (1-new_lr)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = new_lr*bellman_update
		self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b

	def update(self, episode: list, lr: float=.05) -> None:
		# cumulative_r = 0#self.v_table[tuple(episode[-1][2])]
		#shorten trajectory?
		lr = min(lr, .95)
		for i in range(len(episode)): 
			frame = episode[-i]
			state, desired_goal, achieved_goal, action, next_state, reward = frame
			assert state.shape == next_state.shape

			for i in range(repeat_num):
				if np.random.rand() < 1/k:
					if USE_Q: self._update_pure_q(state, desired_goal, achieved_goal, action, next_state, reward, lr)
					if USE_HER: self._update_q(state, desired_goal, achieved_goal, action, next_state, reward, lr)
					if USE_USHER: 
						self._update_off_target_usher_q(state, desired_goal, achieved_goal, desired_goal, action, next_state, lr, p=1)

				# for _ in range(k):
					# rand_i = np.random.randint(len(episode) - i) + i
				if USE_HER:
					rand_i: int = np.random.randint(low=len(episode) - i - 1, high=len(episode)) 
					goal = episode[rand_i][2] #Achieved goal
					self._update_q(state, goal, achieved_goal, action, next_state, reward, lr)

				if USE_USHER:
					rand_i = np.random.geometric(1-gamma, size=1)[0]
					#Could just be np.random.geometric(1-gamma), but this makes mypy happy

					# goal = episode[rand_i][2] if rand_i < i else desired_goal
					goal = episode[len(episode) - i + rand_i][2] if rand_i < i else desired_goal
					if (goal == desired_goal).all():
						self._update_probability(state, goal, desired_goal, achieved_goal, action, next_state, lr)
						# self._update_off_target_usher_q(state, goal, desired_goal, action, next_state, lr, p=gamma**(len(episode) - i))
						self._update_off_target_usher_q(state, goal, desired_goal, achieved_goal, action, next_state, lr, p=gamma**i)





index_to_action = {
	0: np.array([1,0]),
	1: np.array([-1,0]),
	2: np.array([0,1]),
	3: np.array([0,-1]),
	4: np.array([0,0])
}


index_to_string = {
	0: "right",
	1: "left",
	2: "down",
	3: "up",
	4: "stay"
}



# # @typechecked
# def observe_transition(env, state: State, q: Q, policy: Callable) -> Transition:
# 	action = policy(state)
# 	# pdb.set_trace()
# 	env_action = index_to_action[action]
# 	obs, reward, done, info = env.step(state, env_action)
# 	dg=obs['desired_goal']
# 	ag=obs['achieved_goal']
# 	next_state=obs['observation']
# 	assert state.shape == next_state.shape
# 	rv = (state, dg, ag, action, next_state, reward)
# 	if state[-1] == 1:
# 		assert (state == next_state).all()
# 		assert (state[:2] == ag).all()
# 	return rv


# # @typechecked
# def observe_episode(env, q: Q, policy: Callable) -> List[Tuple]:
# 	obs = env.reset()
# 	state = obs['state']	
# 	steps = []
# 	for _ in range(traj_steps):
# 		obs = observe_transition(env, state, q, policy)
# 		state = obs[0]
# 		steps.append(obs)
# 	return steps

# # @typechecked
def observe_transition(env, state: State, q: Q, policy: Callable) -> Transition:
	state = env.get_state()
	# env.set_state(state)
	action = policy(state)
	env_action = index_to_action[action]
	obs, reward, done, info = env.step(env_action)
	dg=obs['desired_goal']
	ag=obs['achieved_goal']
	next_state=obs['observation']
	assert state.shape == next_state.shape
	rv = (state, dg, ag, action, next_state, reward)
	if state[-1] == 1:
		assert (state == next_state).all()
		assert (state[:2] == ag).all()
	return rv


# # @typechecked
# def observe_episode(env, q: Q, policy: Callable) -> List[Tuple]:
# 	env.reset()
# 	steps = []
# 	for _ in range(traj_steps):
# 		steps.append(observe_transition(env, q, policy))
# 	return steps

# @typechecked
def observe_episode(env, q: Q, policy: Callable, test: bool = False) -> List[Tuple]:
	obs = env.reset()
	state = obs['state']	
	steps = []
	n = traj_steps if not test else traj_steps*5
	for _ in range(traj_steps):
		obs = observe_transition(env, state, q, policy)
		state = obs[0]
		steps.append(obs)
	return steps

# @typechecked
def collect_episodes(env, q, s, goal, n=50):
	eps = []
	for _ in range(n):
		eps.append(observe_episode(env, q, lambda s: q.argmax_action(s, goal)))

	return eps


def learn_q_function():
	compute_reward = env.compute_reward
	default_goal = env.new_goal
	q = Q(env.size, compute_reward, default_goal)
	ave_r = 0
	ave_q_r = 0
	ave_usher_r = 0

	if DISPLAY:	display_init(env, q)
	iterations = []
	her_vals = []
	usher_vals = []
	q_vals = []
	ave_r_vals = []
	ave_q_r_vals = []
	ave_usher_r_vals = []
	
	get_ave_r = lambda ep: (1-gamma)*sum([gamma**i*ep[i][-1] for i in range(len(ep))])

	for episode in range(episodes):
		if DISPLAY:	draw_grid(env, q)
		state = env.reset()['observation']
		power = .7
		# power = .95
		lr = (2**(-100*episode/episodes) + base_lr/(episode**power/100+1))
		# lr = .1*base_lr*episodes/(.1*episodes + episode)
		# lr = .05
		if episode%record_num == 0: 
			print("---------------------------")
			print(f"Episode {episode} of {episodes}")
			# [q.update_v(ep) for ep in eps]
			# ave_lr = 1/(episodes+1)#2**(-50*episode/episodes)
			ave_lr = min(traj_steps*(2**(-100*episode/episodes) + base_lr/(episode**power/100+1)), .9)


			if USE_HER:
				her_val = q.q_table[tuple(state) + tuple(default_goal)].max()
				eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal, policy="HER"), test=True) for _ in range(50)]
				ave_r = ave_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in eps])
				her_vals.append(her_val)
				ave_r_vals.append(ave_r)	
				print(f"HER q value: \t\t\t{her_val}")
				# print(f"Average HER reward: \t\t{ave_r}")

			if USE_Q: 
				q_val = q.pure_q_table[tuple(state) + tuple(default_goal)].max()
				q_eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal, policy="Q"), test=True) for _ in range(50)]
				ave_q_r = ave_q_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in q_eps])
				ave_q_r_vals.append(ave_q_r)
				q_vals.append(q_val)	
				print(f"Pure q value: \t\t\t{q_val}")
				# print(f"Average Q return: \t\t{ave_q_r}")

			if USE_USHER: 
				usher_val = q.usher_q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()
				usher_eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal, policy="USHER"), test=True) for _ in range(50)]
				ave_usher_r = ave_usher_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in usher_eps])
				ave_usher_r_vals.append(ave_usher_r)	
				usher_vals.append(usher_val)
				print(f"USHER weighted q value: \t{usher_val}")
				# print(f"Average USHER return: \t\t{ave_usher_r}")

			print()

			if USE_HER:
				print(f"Average HER reward: \t\t{ave_r}")

			if USE_Q: 
				print(f"Average Q return: \t\t{ave_q_r}")

			if USE_USHER: 
				print(f"Average USHER return: \t\t{ave_usher_r}")

			print()

			if USE_HER:
				action = index_to_string[q.argmax_action(state, default_goal, policy="HER")]
				print(f"HER action: \t\t{action}")

			if USE_Q: 
				action = index_to_string[q.argmax_action(state, default_goal, policy="Q")]
				print(f"Q action: \t\t{action}")

			if USE_USHER: 
				action = index_to_string[q.argmax_action(state, default_goal, policy="USHER")]
				print(f"USHER action: \t\t{action}")

			iterations.append(episode*record_num*traj_steps)

			#-------------------------------------------------------------------		
			# usher_eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal, policy="USHER")) for _ in range(50)]
			# ave_usher_r = ave_usher_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in usher_eps])
			
			# print(f"Average USHER return: \t\t{ave_usher_r}")


			# ave_usher_r_vals.append(ave_usher_r)	


		ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal, policy="HER"))
		q.update(ep, lr=lr)
		ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal, policy="Q"))
		q.update(ep, lr=lr)
		# ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal, policy="HER"))
		# q.update(ep, lr=lr)
		ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal, policy="USHER"))
		q.update(ep, lr=lr)


	plt.title("Rewards from each method")
	plt.plot(iterations, ave_r_vals,label="HER reward")
	plt.plot(iterations, ave_q_r_vals,label="Q learning reward")
	plt.plot(iterations, ave_usher_r_vals,label="USHER reward")
	plt.xlabel("Interactions")
	plt.ylabel("Value/Reward")
	plt.legend()
	try: 
		plt.savefig(f"./figures/{DOORS}-door_reward.png")
	except: 
		print("failed to save")
	# plt.show()
	plt.close()


	plt.title("Value and rewards from each method")
	plt.plot(iterations, her_vals, 	label="HER value")
	plt.plot(iterations, q_vals, 	label="Q value")
	plt.plot(iterations, usher_vals,label="USHER value")
	plt.plot(iterations, ave_r_vals,label="HER reward")
	plt.plot(iterations, ave_q_r_vals,label="Q learning reward")
	plt.plot(iterations, ave_usher_r_vals,label="USHER reward")
	plt.xlabel("Interactions")
	plt.ylabel("Value/Reward")
	plt.legend()
	try: 
		plt.savefig(f"./figures/{DOORS}-door_reward_and_value.png")
	except: 
		print("failed to save")
	# plt.show()
	plt.close()

	her_biases = [v-r for r, v in zip(ave_r_vals, her_vals)]
	q_biases = [v-r for r, v in zip(ave_q_r_vals, q_vals)]
	usher_biases = [v-r for r, v in zip(ave_usher_r_vals, usher_vals)]

	plt.title("Biases of each method")
	plt.plot(iterations, her_biases,label="HER bias")
	plt.plot(iterations, q_biases,label="Q bias")
	plt.plot(iterations, usher_biases,label="USHER bias")
	plt.xlabel("Interactions")
	plt.ylabel("Bias")
	plt.legend()
	try: 
		plt.savefig(f"./figures/{DOORS}-door_bias.png")
	except: 
		print("failed to save")
	# plt.show()
	plt.close()



learn_q_function()
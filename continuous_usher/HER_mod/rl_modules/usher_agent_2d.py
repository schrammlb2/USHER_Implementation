import torch
import os
from datetime import datetime
import numpy as np
import itertools
from mpi4py import MPI

from HER_mod.mpi_utils.mpi_utils import sync_networks, sync_grads
# from HER_mod.mpi_utils.normalizer import normalizer
from HER.mpi_utils.normalizer import normalizer

from HER_mod.her_modules.her import her_sampler

from HER_mod.rl_modules.replay_buffer import replay_buffer
from HER_mod.rl_modules.models import sac_actor as actor
from HER_mod.rl_modules.models import T_conditioned_ratio_critic as critic
from HER_mod.rl_modules.hyperparams import POS_LIMIT

from loss_functions import usher_loss




"""

implementation of USHER for low-dimensional environments. We use this implementation for all experiments except Fetch

"""
DTYPE = torch.float32

POLYAK_SCALE = .0
train_on_target = False
train_on_target = True


class ValueEstimator:
  def __init__(self, env_params, args):
    self.args = args
    self.env_params = env_params

    self.double_q = False

    self.critic_1 = critic(env_params)
    sync_networks(self.critic_1)
    self.critic_target_1 = critic(env_params)
    self.critic_target_1.load_state_dict(self.critic_1.state_dict())    
    if self.double_q: 
        self.critic_2 = critic(env_params)
        sync_networks(self.critic_2)
        self.critic_target_2 = critic(env_params)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())    
        self.critics_optimiser = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.args.lr_critic, weight_decay=.004)
    else: 
        self.critics_optimiser = torch.optim.Adam(list(self.critic_1.parameters()))# + list(self.critic_2.parameters()), lr=self.args.lr_critic, weight_decay=.004)

    if self.args.cuda:
        self.critic_1.cuda()
        self.critic_2.cuda()
        self.critic_target_1.cuda()
        self.critic_target_2.cuda()


  def min_critic(self, state, T, action):
    tc1 = self.critic_1(state, T, action)
    if not self.double_q:
        return tc1
    tc2 = self.critic_2(state, T, action)
    return torch.min(tc1, tc2)

  def min_critic_target(self, state, T, action):
    tc1 = self.critic_target_1(state, T, action)
    if not self.double_q:
        return tc1
    tc2 = self.critic_target_2(state, T, action)
    return torch.min(tc1, tc2)

  # def t_to_r(self, t):
  #   min_r = -1 / (1 - self.args.gamma)
  #   return min_r*(1-self.args.gamma**t)
    
  def q_loss(self, preprocced_tuple, actor, transitions):
    inputs_next_norm_tensor, inputs_norm_tensor_pol, inputs_next_norm_tensor_pol = preprocced_tuple
    low_range = False
    # low_range = True
    delta_p = False
    if delta_p: assert low_range == False
    split_p_evals = True
    split_p_evals = False
    with torch.no_grad():        
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        exact_goal_tensor = torch.tensor(transitions['exact_goal'], dtype=torch.float32) 
        t = torch.tensor(transitions['t_remaining'], dtype=torch.float32) 
        her_used = torch.tensor(transitions['her_used'], dtype=torch.float32) 
        map_t = lambda t: -1 + 2*t/self.env_params['max_timesteps']
        # do the normalization
        # concatenate the stuffs
        actions_next = actor(inputs_next_norm_tensor, deterministic=True)
        shape = actions_next.shape
        TARGET_ACTION_NOISE = .1
        TARGET_ACTION_NOISE_CLIP = .25
        actions_next = torch.clamp(actions_next, min=-1, max=1)
        # clip the q value
        clip_return = 1 / (1 - self.args.gamma)

        q_next_value, p_next_value = self.critic_target_1(inputs_next_norm_tensor_pol, map_t(t-1), actions_next, return_p=True)
        q_next_value = q_next_value.detach()
        if self.args.non_terminal_goals: 
            target_q_value = r_tensor + self.args.gamma * q_next_value 
        else: 
            target_q_value = r_tensor + self.args.gamma * q_next_value * (-r_tensor)
        target_q_value = target_q_value.detach()

        target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        p_next_value = p_next_value.detach()
        target_p_value = 1/t*((1-her_used)*1/(self.args.replay_k + 1) + exact_goal_tensor*self.args.replay_k/(self.args.replay_k + 1)) + p_next_value*(t-1)/t
        target_p_value = target_p_value.detach()
        

    q0, p0 = self.critic_1(inputs_norm_tensor_pol, map_t(t), actions_tensor, return_p=True)

    if self.args.apply_ratio: 
        true_c = self.args.ratio_offset
        p_num = p0.detach()
        p_denom =  p_next_value.detach()
        
        method = "indep_goal"
        if method == "indep_goal": 
            _, indep_goal_input = self.get_input_tensor(transitions['obs'], transitions['alt_g'], transitions['policy_g'])
            _, indep_goal_input_next = self.get_input_tensor(transitions['obs'], transitions['alt_g'], transitions['policy_g'])

            q_indep_goal, p_indep_goal = self.critic_1(indep_goal_input, map_t(t), actions_tensor, return_p=True)
            q_indep_goal_next, p_indep_goal_next = self.critic_target_1(indep_goal_input_next, map_t(t), actions_tensor, return_p=True)

            true_c = self.args.ratio_offset
            q_alpha = .1
            p_alpha = .5
            c = true_c
            def indep_w(alpha, clip=False):  
                x = (p_indep_goal.detach() + c)/(alpha*p_indep_goal.detach() + (1-alpha)*p_indep_goal_next.detach() + c)*her_used + (1-her_used)
                if clip: 
                    return alpha*torch.clamp(x, 1/clip_scale, clip_scale)
                else: 
                    return alpha*x

            def her_w(alpha, clip=False):  
                x = (p0.detach() + c)/(alpha*p0.detach() + (1-alpha)*p_next_value.detach() + c)*her_used + (1-her_used)
                if clip: 
                    return (1-alpha)*torch.clamp(x, 1/clip_scale, clip_scale)
                else: 
                    return (1-alpha)*x

            clip_scale = 1+self.args.ratio_clip
            clip = lambda x: torch.clamp(x, 1/clip_scale, clip_scale)

            _, on_policy_input = self.get_input_tensor(transitions['obs'], transitions['ag_next'], transitions['policy_g'])
            _ , realized_p = self.critic_1(on_policy_input, map_t(t), actions_tensor, return_p=True)

            target_q_indep_goal_value = torch.tensor(transitions['alt_r'], dtype=torch.float32)  + self.args.gamma * q_indep_goal_next
            q_loss = (her_w(q_alpha, clip=True)*(target_q_value - q0).pow(2)).mean()
            q_loss = q_loss + (indep_w(q_alpha, clip=True)*(target_q_indep_goal_value - q_indep_goal).pow(2)).mean()
            p_loss = (her_w(p_alpha)*(((p0).pow(2)  - (t-1)/t*(p0*target_p_value))*her_used)).mean()
            p_loss = p_loss + (indep_w(p_alpha)*her_used*((p_indep_goal).pow(2)- (t-1)/t*p_indep_goal*p_indep_goal_next)).mean() - 2*(realized_p/t).mean()/100

            critic_loss = q_loss + p_loss

    else: 
        true_ratio = 1
        critic_loss = (target_q_value - q0).pow(2).mean() 
                                                        
    return critic_loss


  def _soft_update_target_network(self, target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

  def update(self, preprocced_tuple, actor, transitions):
    # Update Q-functions by one step of gradient descent
    self.critics_optimiser.zero_grad()
    self.q_loss(preprocced_tuple, actor, transitions).backward()
    self.critics_optimiser.step()

    # Update target value network
    self._soft_update_target_network(self.critic_target_1, self.critic_1)
    if self.double_q: 
        self._soft_update_target_network(self.critic_target_2, self.critic_2)


class ddpg_agent:
    def __init__(self, args, env, env_params, vel_goal=True):
        self.global_count = 0

        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic = ValueEstimator(env_params, args)
        self.critic.get_input_tensor = self.get_input_tensor
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.actor_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, 
            self.env.compute_reward, args.gamma, args.two_goal, False)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions, ddpg_sample_func=self.her_module.sample_ddpg_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if args.apply_ratio: 
            agent_name = "usher"
        elif args.two_goal: 
            agent_name = "two-goal"
        elif args.replay_k == 0:
            agent_name = "q-learning"
        else:
            agent_name = "her"
        key = f"name_{args.env_name}__noise_{args.action_noise}__agent_{agent_name}.txt"
        self.recording_path = "logging/recordings/" + key
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

            with open(self.recording_path, "w") as file: 
                file.write("")


        self.normalize = True
        self.vel_goal = vel_goal


    def learn(self, hooks=[], epochs=None):
        """
        train the network

        """
        # start to collect samples
        epochs = self.args.n_epochs if epochs == None else epochs
        for epoch in range(epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                mb_col = []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    ep_col = []
                    # reset the environment
                    observation = self.env.reset()

                    random_next_goal = (np.random.rand(2)*2-1)*POS_LIMIT

                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, done, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        ep_col.append(False)
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_col.append(ep_col)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_col = np.array(mb_col)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_col])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions, mb_col])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
            # start to do the evaluation
            self.actor_network.eval()
            ev = self._eval_agent()
            self.actor_network.train()
            success_rate, reward, value = ev['success_rate'], ev['reward_rate'], ev['value_rate']

            import time
            time.sleep(np.random.rand()*5)
            with open(self.recording_path, "a") as file: 
                file.write(f"{epoch}, {success_rate:.3f}, {reward:.3f}, {value:.3f}\n")


            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f'[{datetime.now()}] epoch is: {epoch}, '
                    f'eval success rate is: {success_rate:.3f}, '
                    f'average reward is: {reward:.3f}, '
                    f'average value is: {value:.3f}')
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g, gpi=None):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        if gpi is not None: 
            gpi_norm = self.g_norm.normalize(g)
            inputs = np.concatenate([obs_norm, g_norm, gpi_norm])
        else:
            inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        # pass
        mb_obs, mb_ag, mb_g, mb_actions, mb_col = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'col': mb_col, 
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.actor_network.set_normalizers(self.o_norm.get_torch_normalizer(), self.g_norm.get_torch_normalizer())

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def get_input_tensor(self, obs, goal, policy_goal):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(goal)
        pol_g_norm = self.g_norm.normalize(policy_goal)

        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        inputs_norm_pol = np.concatenate([obs_norm, g_norm, pol_g_norm], axis=1)

        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_norm_tensor_pol = torch.tensor(inputs_norm_pol, dtype=torch.float32)

        return inputs_norm_tensor, inputs_norm_tensor_pol

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, pol_g = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['policy_g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['policy_g'] = self._preproc_og(o, pol_g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        pol_g_norm = self.g_norm.normalize(transitions['policy_g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        inputs_norm_pol = np.concatenate([obs_norm, g_norm, pol_g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        inputs_next_norm_pol = np.concatenate([obs_next_norm, g_next_norm, pol_g_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        inputs_norm_tensor_pol = torch.tensor(inputs_norm_pol, dtype=torch.float32)
        inputs_next_norm_tensor_pol = torch.tensor(inputs_next_norm_pol, dtype=torch.float32)

        duplicated_g_input = torch.tensor(np.concatenate([obs_norm, g_norm, g_norm], axis=1), dtype=torch.float32)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        t = torch.tensor(transitions['t_remaining'], dtype=torch.float32) 
        map_t = lambda t: -1 + 2*t/self.env_params['max_timesteps']
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        self.global_count += 1
        use_random_actor = True
        if self.global_count % 2 == 0:
            actions_real, log_prob = self.actor_network(inputs_norm_tensor, with_logprob=True)
            if train_on_target: 
                actor_loss = -self.critic.min_critic_target(duplicated_g_input, map_t(t), actions_real).mean()
            else: 
                actor_loss = -self.critic.min_critic(duplicated_g_input, map_t(t), actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            if use_random_actor:
                actor_loss += self.args.entropy_regularization*log_prob.mean()
            else: 
                actor_loss -= self.args.entropy_regularization*log_prob.mean()
            # start to update the network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor_network)
            self.actor_optim.step()
        # update the critic_network
        tup = (inputs_next_norm_tensor, inputs_norm_tensor_pol, inputs_next_norm_tensor_pol)
        self.critic.update(tup, self.actor_target_network, transitions)

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_reward_rate = []
        total_value_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            total_r = 0
            total_value = 0


            with torch.no_grad():
                input_tensor = self._preproc_inputs(obs, g)
                pi = self.actor_network(input_tensor)
                actions_tensor = pi.detach().cpu()
                actions = actions_tensor.numpy().squeeze(axis=0)
                input_tensor_pol = self._preproc_inputs(obs, g, gpi=g)
                t = torch.tensor([[1]])
                value = self.critic.min_critic(input_tensor_pol, t, actions_tensor).mean().numpy().squeeze()
                total_value += value

            #Use enough time steps to let the total discounted reward approach the value
            for t in range(int(3/(1-self.args.gamma))):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    actions = pi.detach().cpu().numpy().squeeze(axis=0)

                observation_new, r, _, info = self.env.step(actions)
                total_r += r*self.args.gamma**t
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])

            total_success_rate.append(per_success_rate)
            total_reward_rate.append(total_r)
            total_value_rate.append(total_value)

        total_success_rate = np.array(total_success_rate)
        total_reward_rate = np.array(total_reward_rate)
        total_value_rate = np.array(total_value_rate)

        local_success_rate = np.mean(total_success_rate[:, -1])
        local_reward_rate = np.mean(total_reward_rate)
        local_value_rate = np.mean(total_value_rate)
        
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_reward_rate = MPI.COMM_WORLD.allreduce(local_reward_rate, op=MPI.SUM)
        global_value_rate = MPI.COMM_WORLD.allreduce(local_value_rate, op=MPI.SUM)

        return {
            'success_rate': global_success_rate / MPI.COMM_WORLD.Get_size(), 
            'reward_rate': global_reward_rate / MPI.COMM_WORLD.Get_size(), 
            'value_rate': global_value_rate / MPI.COMM_WORLD.Get_size(), 
            }


    def render_run(self, i):
        per_success_rate = []
        observation = self.env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        from display import display_init, draw_grid
        env_name = self.args.env_name
        display_init(self.env.env)
        draw_grid(self.env.env, plot_agent = True, filename=f"{env_name}_{0}")

        for t in range(int(3/(1-self.args.gamma))):
            draw_grid(self.env.env, plot_agent = True, filename=f"{env_name}_{i}_{t}")
            with torch.no_grad():
                input_tensor = self._preproc_inputs(obs, g)
                pi = self.actor_network(input_tensor)
                actions = pi.detach().cpu().numpy().squeeze(axis=0)

            observation_new, r, _, info = self.env.step(actions)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
        final_t = t + 1
        draw_grid(self.env.env, plot_agent = True, filename=f"{env_name}_{i}_{final_t}")


    def record_run(self, i):
        per_success_rate = []
        observation = self.env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        from display import display_init, draw_grid
        import pickle
        env_name = self.args.env_name

        def dump(env, filename): 
            #Env is not pickleable, so copy out stuff thats important for display()
            obj = Object()
            obj.grid = env.grid
            obj.state = env.state
            obj.start = env.start
            obj.goal = env.goal
            obj.width = env.width
            obj.length = env.length
            obj.path = env.path

            obj.env = Object()
            obj.env.env = env.env.env
            obj.env.state_to_goal = obj.env.env.state_to_goal
            obj.env.state_to_obs = obj.env.env.state_to_obs
            obj.env.state_to_rot = obj.env.env.state_to_rot
            pickle.dump(obj, open(filename, "wb"))

        for t in range(int(3/(1-self.args.gamma))):            
            dump(self.env.env, f"logging/path_recordings/{env_name}_{i}_{t}")
            with torch.no_grad():
                input_tensor = self._preproc_inputs(obs, g)
                pi = self.actor_network(input_tensor)
                actions = pi.detach().cpu().numpy().squeeze(axis=0)

            observation_new, r, _, info = self.env.step(actions)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
        dump(self.env.env, f"{env_name}_{i}_{t}")

class Object(object):  pass
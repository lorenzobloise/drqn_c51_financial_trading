import copy
import torch
import torch.nn.init as weight_init
from agents.trading_agent import TradingAgent
from models.drqn_c51 import DRQN_C51
import os
import random
import numpy as np
from memory import Transition, ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent_C51(TradingAgent):

	def __init__(self, args, state_size=14, epsilon_greedy=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 3
		self.memory = ReplayMemory(args.replay_memory_size)
		self.inventory = []
		self.epsilon_greedy = epsilon_greedy
		self.T = args.T
		self.gamma = args.gamma
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999
		self.batch_size = args.batch_size
		# Added parameters
		self.num_atoms = 51
		self.v_min = -10
		self.v_max = 10
		self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
		self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)
		if os.path.exists('models/target_model'):
			self.policy_net = torch.load('models/policy_model', map_location=device)
			self.target_net = torch.load('models/target_model', map_location=device)
		else:
			self.policy_net = DRQN_C51(state_size, self.action_size, self.num_atoms).to(device)
			self.target_net = DRQN_C51(state_size, self.action_size, self.num_atoms).to(device)
			for param_p in self.policy_net.parameters():
				weight_init.normal_(param_p)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.lr)

	def act(self, state):
		if self.epsilon_greedy and np.random.rand() <= self.epsilon:
			self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
			return random.randrange(self.action_size) - 1
		# IMPLEMENTATION OF C51 CATEGORICAL DQN
		# PRESENTED IN 'A DISTRIBUTIONAL PERSPECTIVE ON REINFORCEMENTE LEARNING' BY BELLAMARE ET AL., 2017
		# Part 1: compute the Q-value and determine the best action to take
		tensor = torch.FloatTensor(state).to(device)
		tensor = tensor.unsqueeze(0)
		action_value_dist = self.target_net(tensor) # (batch_size*seq_len, num_actions, num_atoms)
		action_values = torch.sum(action_value_dist * self.support, dim=-1) # (batch_size * seq_len, num_actions)
		ret = action_values[-1]
		ret = ret.detach().cpu().numpy()
		action = np.argmax(ret)
		if self.epsilon_greedy:
			self.epsilon = max(self.epsilon_min,self.epsilon*self.epsilon_decay)
		return action-1

	def store(self, state, actions, new_states, rewards, action, step):
		if step < 1000: # soft update
			for n in range(len(actions)):
				self.memory.push(state, actions[n], new_states[n], rewards[n])
		else:
			for n in range(len(actions)):
				if actions[n] == action:
					self.memory.push(state, actions[n], new_states[n], rewards[n])
					break

	def optimize(self, step):
		if len(self.memory) < self.batch_size * 10:
			return
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))
		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state = torch.FloatTensor(np.array(batch.next_state)).to(device)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
		non_final_next_states = torch.cat([s for s in next_state if s is not None])
		state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
		action_batch = torch.LongTensor(torch.add(torch.tensor(batch.action), torch.tensor(1))).to(device)
		reward_batch = torch.FloatTensor(batch.reward).to(device)
		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		# state batch [32, 96, 14]
		policy_net_output = self.policy_net(state_batch) # [3072, 3, 51]
		l = policy_net_output.size(0)
		state_action_values = policy_net_output[95:l:96] # [32, 3, 51]
		action_batch = action_batch.view((-1,1,1)) # [32] -> [32, 1, 1]
		if state_action_values.size(0) == 0:
			print(f"Errore: state_action_values è vuoto: {state_action_values.shape}.")
			return
		state_action_values = state_action_values.gather(1, action_batch) # [32, 1, 1]
		state_action_values = state_action_values.squeeze(-1) # [32, 1]
		# IMPLEMENTATION OF C51 CATEGORICAL DQN
		# PRESENTED IN 'A DISTRIBUTIONAL PERSPECTIVE ON REINFORCEMENTE LEARNING' BY BELLAMARE ET AL., 2017
		# Part 2: project onto the support and distribute probability of Tz
		with torch.no_grad():
			prob = self.target_net(next_state) # (batch_size*seq_len, num_actions, num_atoms)
			prob = prob[95:l:96]
			q_value = (prob * self.support)
			q_value = q_value.sum(dim=2)
			next_action = q_value.argmax(1)
			next_dist = prob.gather(1, next_action.view(-1,1,1).expand(-1,-1,self.num_atoms)).squeeze(1)
			t_z = reward_batch.unsqueeze(1) + self.gamma*self.support.unsqueeze(0)
			t_z = t_z.clamp(min=self.v_min,max=self.v_max)
			b = (t_z-self.v_min)/self.delta_z
			l = b.floor().long()
			u = b.ceil().long()
			m = torch.zeros(self.batch_size,self.num_atoms,device=device)
			offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(device) # (1)
			m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
			m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
		loss = -(m*torch.log(state_action_values)).sum(-1).mean()
		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()
		if step % self.T == 0:
			gamma = 0.001
			param_before = copy.deepcopy(self.target_net)
			target_update = copy.deepcopy(self.target_net.state_dict())
			for k in target_update.keys():
				target_update[k] = self.target_net.state_dict()[k] * (1 - gamma) + self.policy_net.state_dict()[
					k] * gamma
			self.target_net.load_state_dict(target_update)
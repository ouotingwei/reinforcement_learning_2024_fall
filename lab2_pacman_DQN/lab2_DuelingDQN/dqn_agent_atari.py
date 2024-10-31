import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model_dueling import AtariNetDQN
import gym
import random

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = atari_preprocessing.AtariPreprocessing(self.env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.env = FrameStack(self.env, 4)


		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		#self.test_env = gym.make(config["env_id"], render_mode="human")
		self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		# random -> prevent actions track into local minimum
		if random.random() < epsilon:
			# random exploration
			action = self.env.action_space.sample()
		else:
			# epsilon-greedy
			# obervation -> tensor
			observation_tensor = torch.tensor(np.array(observation), dtype=torch.float, device=self.device).unsqueeze(0)

			# in the inference state, there's no needed to calculate the gradiant
			with torch.no_grad():  
				# calculate the Q-value with the behavior net
				action_values = self.behavior_net(observation_tensor) 
			
			# find the best action (largest Q-value) as the action 
			action = torch.argmax(action_values).item()

		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		# 1. get Q(s, a) from the behavior network
		q_value = self.behavior_net(state).gather(1, action.long())

		# 2. get max_a Q(s, a) from the target network
		with torch.no_grad():
			# Behavior network selects the action with max Q-value for the next state
			next_action = self.behavior_net(next_state).max(1)[1].unsqueeze(1)

			# Target network calculates the Q-value for the action selected by the behavior network
			q_next = self.target_net(next_state).gather(1, next_action) 
			
			# calculate Q_target = r + gamma * max_a Q(s',a)
			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)
        
		# 4. calculate loss between Q(s,a) and Q_target
		criterion = torch.nn.MSELoss()
		loss = criterion(q_value, q_target)

		# log the loss to TensorBoard
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		# 5. update behavior net
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
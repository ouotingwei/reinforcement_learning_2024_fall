import numpy as np
import torch
import torch.nn as nn
import gym
from lab2_DQN.dqn_agent_atari import AtariDQNAgent
from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack

# env = gym.make("ALE/Enduro-v5", render_mode="human")

# if you don't have GUI, you can use the following code to record video
# env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
# env = gym.wrappers.RecordVideo(env, 'video')

config = {
    "gpu": True,
    "training_steps": 1e8,
    "gamma": 0.99,
    "batch_size": 32,
    # "batch_size": 8,
    "eps_min": 0.1,
    "warmup_steps": 20000,
    "eps_decay": 1000000,
    "eval_epsilon": 0.01,
    "replay_buffer_capacity": 100000,
    # "logdir": 'log/DQN/MsPacman/',
    "logdir": 'log/DQN/Enduro/',
    "update_freq": 4,
    "update_target_freq": 10000,
    "learning_rate": 0.0000625,
    "eval_interval": 100,
    "eval_episode": 5,
    # "env_id": 'ALE/MsPacman-v5',
    "env_id": 'ALE/Enduro-v5',
}

# env = gym.make(config["env_id"], render_mode="human")
env = gym.make(config["env_id"], render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'video')
env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
env = FrameStack(env, 4)

# Load the model
agent = AtariDQNAgent(config)
agent.behavior_net.load_state_dict(torch.load('weight_and_data/lab2/pcaman_DQN2_yes/model_17207753_3774.pth'))
# agent.behavior_net.load_state_dict(torch.load('log/DQN/MsPacman_final/model_97811074_5776.pth'))
agent.behavior_net.eval()

observation, info = env.reset()
total_reward = 0

while True:
    # env.render()
    action = agent.decide_agent_actions(observation, 0.0, env.action_space)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Total reward: {}".format(total_reward))
env.close()
import numpy as np
import torch
import torch.nn as nn
import gym
from ppo_agent_atari import AtariPPOAgent

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack

config = {
    "gpu": True,
    "training_steps": 1e8,
    "update_sample_count": 10000,
    "discount_factor_gamma": 0.99,
    "discount_factor_lambda": 0.95,
    "clip_epsilon": 0.2,
    "max_gradient_norm": 0.5,
    "batch_size": 128,
    #"batch_size": 1024,
    "logdir": '/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data',
    "update_ppo_epoch": 3,
    "learning_rate": 2.5e-5,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "horizon": 128,
    "env_id": 'ALE/Enduro-v5',
    "eval_interval": 100,
    "eval_episode": 3,
}

env = gym.make(config["env_id"], render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'video')
env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
env = FrameStack(env, 4)

# Load the model
agent = AtariPPOAgent(config)
agent.load(torch.load('/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab3_smaller_training_rate/model_23238608_1977.pth'))
agent.evaluate()

observation, info = env.reset()
total_reward = 0

while True:
    # env.render()
    action, value, logp_pi = agent.decide_agent_actions(observation, 0.0, env.action_space)
    next_observation, reward, terminate, truncate, info = env.step(action)
    # total_reward += reward
    # if terminated or truncated:
    #     break

    obs = {}
    obs["observation_2d"] = np.asarray(observation, dtype=np.float32)

    total_reward += reward
    
    if terminate or truncate:
        break

print("Total reward: {}".format(total_reward))
env.close()
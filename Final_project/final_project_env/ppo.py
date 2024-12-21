# Ensure you have installed racecar_gym so you can directly import the env
from racecar_gym.env import RaceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation


import numpy as np
import os
    
def EnvWrapper():
    env = RaceEnv(
        scenario='austria_competition_collisionStop',  # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
        render_mode='rgb_array_birds_eye',
        reset_when_collision = True  # Only works for 'austria_competition' and 'austria_competition collisionStop'
    )
    #print("Initial observation space shape:", env.observation_space.shape)

    # Change channel to last if needed
    env = ChannelLastObservation(env)
    #print("Observation space shape after ChannelLastObservation:", env.observation_space.shape)

    # Convert to grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    #print("Observation space shape after GrayScaleObservation:", env.observation_space.shape)

    # Resize observation
    env = ResizeObservation(env, (64, 64))
    # print("Observation space shape after ResizeObservation:", env.observation_space.shape)

    # print("ok")
    return env


class ChannelLastObservation(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(ChannelLastObservation, self).__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[1], obs_shape[2], obs_shape[0])
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, (1, 2, 0))


class PPO_callback(BaseCallback):
    def __init__(self, check_frequency, save_path, verbose=True):
        super(PPO_callback, self).__init__(verbose)
        self.check_frequency = check_frequency
        self.save_path = save_path
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_frequency == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return super()._on_step()


if __name__ == '__main__':
    print('gooooooooooooooooooooooooooooo')
    Save_dir = '/workingdirectory/model_save_ppo13'
    Check_freq = 1000
    Process = 16
    # env = EnvWrapper()
    
    env = SubprocVecEnv([lambda: EnvWrapper() for _ in range(Process)])
    env = VecFrameStack(env, n_stack=10, channels_order='last')
    env = VecMonitor(env)

    callback = PPO_callback(check_frequency=Check_freq, save_path = Save_dir)

    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        tensorboard_log=Save_dir,
        device='cuda',
        use_sde=True,
        batch_size=128, 
        n_steps=1024,
        n_epochs=10,
        learning_rate=1e-4,  
        clip_range=0.2,
        ent_coef=0.01
    )

    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True)
from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 1024,
		"logdir": '/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab3_test',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-5,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/MsPacman-v5',
		"eval_interval": 100,
		"eval_episode": 3,
	}

	agent = AtariPPOAgent(config)
	#agent.load("/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab3_1024bat_smaller_training_rate/model_25145984_2308.pth")
	agent.train()
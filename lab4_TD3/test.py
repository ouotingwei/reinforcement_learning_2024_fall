from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": '',
		"update_freq": 2,
		"eval_interval": 1,
		"eval_episode": 1,
	}

	agent = CarRacingTD3Agent(config)
	agent.load_and_evaluate("/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab4/low_noise/model_815989_842.pth")



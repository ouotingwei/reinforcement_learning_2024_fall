import matplotlib.pyplot as plt

episodes = []
means = []

with open('/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab1/3_mean.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        ep = int(parts[0].split(':')[1])  
        mean = float(parts[1].split(':')[1])  
        episodes.append(ep)
        means.append(mean)

plt.figure(figsize=(10, 6))
plt.plot(episodes, means, marker='', linestyle='-', color='b')
plt.title('Training Process')
plt.xlabel('Episode')
plt.ylabel('Mean Score')
plt.grid(True)
plt.show()
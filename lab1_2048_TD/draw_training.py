import matplotlib.pyplot as plt

tdl_means = []

with open('/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab1/tdlearning_1m.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        #ep = int(parts[0].split(':')[1])  
        mean = float(parts[1].split(':')[1])  
        #episodes.append(ep)
        tdl_means.append(mean)

as_means = []

with open('/home/ee605-wei/reinforcement_learning_2024_fall/weight_and_data/lab1/after_state_1m.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        #ep = int(parts[0].split(':')[1])  
        mean = float(parts[1].split(':')[1])  
        #episodes.append(ep)
        as_means.append(mean)

tdl_episodes = []
for i in range(len(tdl_means)):
    tdl_episodes.append(i+1)

as_episodes = []
for i in range(len(as_means)):
    as_episodes.append(i+1)

plt.figure(figsize=(10, 6))
plt.plot(tdl_episodes, tdl_means, marker='', linestyle='-', color='b')
#plt.plot(as_episodes, as_means, marker='', linestyle='-', color='r')
plt.title('TD Learning for 2048: Training Process')
plt.xlabel('Episode(x1000)')
plt.ylabel('Mean Score')
plt.grid(True)
plt.show()
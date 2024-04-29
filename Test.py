from Bernard import train
from matplotlib import pyplot as plt
import os

n_steps = [1, 5, 20, 50, 100]
rewards = []

for n_steps in n_steps:
    print("starting run at n_steps = ", n_steps)
    rewards.append(train(render=False,
      gamma=0.99,
      lr=0.01,
      betas=(0.9, 0.999),
      entropy_weight=0.01,
      n_steps=n_steps,
      num_episodes=500,
      max_steps=10000,
      print_interval=10,
      method="a2c",
      use_baseline=True))

for i in range(len(n_steps_list)):
    plt.plot(range(len(rewards[i])), rewards[i], label=f'n_steps={n_steps_list[i]}')

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Comparison of Rewards for Different n_steps Settings')
plt.legend()
plt.savefig( os.path.join('Data/models', 'n_steps.png'))
plt.show()
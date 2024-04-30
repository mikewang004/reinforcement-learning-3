from Bernard import train
from matplotlib import pyplot as plt
import os
from scipy.signal import savgol_filter
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_steps_list = [2, 5, 20]
mean_rewards = []

for n_steps in n_steps_list:
    print(f"Starting runs for n_steps = {n_steps} (3 repetitions)")
    run_rewards = []
    for _ in range(3):  # Run the training 3 times
        run_rewards.append(train(render=False,
                                 gamma=0.99,
                                 lr=0.01,
                                 betas=(0.9, 0.999),
                                 entropy_weight=0.01,
                                 n_steps=n_steps,
                                 num_episodes=500,
                                 max_steps=10000,
                                 print_interval=10,
                                 method="a2c",
                                 use_baseline=True,
                                 data_dir="Data/rewards/n_steps{}".format(n_steps)))
    mean_rewards.append(np.mean(run_rewards, axis=0))  # Take the mean of rewards across runs

for i in range(len(n_steps_list)):
    plt.plot(range(len(mean_rewards[i])), savgol_filter(mean_rewards[i], 51, 3), label=f'n_steps={n_steps_list[i]}')

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Comparison of Rewards for Different n_steps Settings (Mean of 3 Runs)')
plt.legend(loc='lower right')
plt.savefig(os.path.join('Data/figures', 'n_steps_mean.png'))
plt.show()
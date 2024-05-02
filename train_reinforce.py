import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np
from Agent import train 
from tqdm import tqdm

lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
#lr_list = [0.001, 0.01]
# Define constants 
gamma = 0.99
betas=(0.9, 0.999)
entropy_weight=0.01
n_steps=10
num_episodes=1000
max_steps=10000
print_interval=10
method = "reinforce"

export_reward_list = []

reps = 6

for lr in lr_list:
    export_reward = np.zeros([reps, int(num_episodes/print_interval)])
    for i in tqdm(range(0, reps), desc="runnin"):
        export_reward[i, :] = train(render = False,
            gamma=gamma,
            lr=lr,
            betas=(0.9, 0.99),
            entropy_weight=entropy_weight,
            n_steps=0,
            num_episodes=num_episodes,
            max_steps=max_steps,
            print_interval=print_interval,
            method = "reinforce")

    avg = np.mean(export_reward, axis = 0)

    export_reward_list.append(avg)

np.savetxt("data/reinforce-lr-test-2.txt", export_reward_list)


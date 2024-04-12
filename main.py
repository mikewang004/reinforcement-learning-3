import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def entropy(p):
    """Calculates entropy as H = - sum_i p_i * log(p_i) while assuming base e. Refer for further details to 
    'Deep Reinforcement Learning' p.111.
    See also the paper by Duan et al. 'Benchmarking Deep Reinforcement Learning for Continuous Control'."""
    return np.sum(p * np.ln(p))

def entropy_optimalisation(theta):
    pass


def main():
    #print('Device is:{}'.format(torch.cuda.get_device_name(0)))
    env = gym.make("LunarLander-v2", render_mode="human")


    observation, info = env.reset()
    env.render()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()
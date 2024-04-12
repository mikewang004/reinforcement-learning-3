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
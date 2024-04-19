import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
from itertools import count
import numpy as np
import os


def create_model(no_obs, no_actions, no_hidden_layers = 8):

        return nn.Sequential(
        nn.Linear(in_features=no_obs,
                  out_features=no_hidden_layers),
        nn.ReLU(),
        nn.Linear(in_features=no_hidden_layers,
                  out_features=no_actions),
    )


def reinforce(model, obs):
    observation_tensor = torch.as_tensor(obs, dtype=torch.float32)
    logits = model(observation_tensor.unsqueeze(dim=1))

    # Categorical will also normalize the logits for us
    return Categorical(logits=logits)

def choose_action(pi):
    a = pi.sample()
    log_probability_action = pi.log_prob(a)

    return int(a.item()), log_probability_action

def get_loss(eps_log_prob, eps_actions_rewards):
    return -(eps_log_prob * eps_actions_rewards).mean()


def train_model(env, model, optimizer, max_timesteps = 2000, eps_timesteps = 200):

    returns = []
    log_probs = []
    rewards = []
    total_time = 0
    while True:
        if total_time > max_timesteps:
            break
        eps_reward = 0
        obs = env.reset()
        for i in range(eps_timesteps):
            policy = reinforce(model, obs[0])
            action, log_probability_action = choose_action(policy)
            obs, reward, done, _, _ = env.step(action)

            eps_reward += reward

            log_probs.append(log_probability_action)

            # Finish the action loop if this episode is done
            if done == True:
                # Add one reward per timestep
                for _ in range(timestep + 1):
                    rewards.append(episode_reward)

                break
        returns.append(eps_reward)
    loss = get_loss(torch.stack(log_probs),torch.as_tensor(rewards, dtype=torch.float32))


    epoch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return np.mean(epoch_returns)

def train(epochs=40) -> None:
    """Train a Vanilla Policy Gradient model on CartPole

    Args:
        epochs (int, optional): The number of epochs to run for. Defaults to 50.
    """

    # Create the Gym Environment
    env = gym.make("LunarLander-v2", render_mode="human")


    # Create the MLP model
    number_observation_features = env.observation_space.shape[0]
    number_actions = env.action_space.n
    model = create_model(number_observation_features, number_actions)

    # Create the optimizer
    optimizer = Adam(model.parameters(), 1e-2)
    # Loop for each epoch
    for epoch in range(epochs):
        average_return = train_model(env, model, optimizer)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))


if __name__ == '__main__':
    train()

def main():
    #print('Device is:{}'.format(torch.cuda.get_device_name(0)))
    env = gym.make("LunarLander-v2", render_mode="human")
    

if __name__ == "__main__":
    main()
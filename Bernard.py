import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np


class ACModel(nn.Module):
    def __init__(self):
        super(ACModel, self).__init__()
        self.actor_common = nn.Linear(8, 128)
        self.actor_action = nn.Linear(128, 4)
        self.critic_value = nn.Linear(128, 1)

        self.log_probs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.actor_common(state))

        state_value = self.critic_value(state)

        action_probs = F.softmax(self.actor_action(state), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.log_probs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculate_entropy(self, prob_distribution):
        dist = Categorical(prob_distribution)
        entropy = dist.entropy().mean()
        return entropy

    def loss(self, gamma=0.99, entropy_weight=0.01):
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        loss = 0
        for log_prob, value, reward in zip(self.log_probs, self.state_values, rewards):
            advantage = reward - value.item()
            loss += (-log_prob * advantage) + F.mse_loss(value, reward)

        entropy = self.calculate_entropy(torch.stack(self.log_probs))
        loss -= entropy_weight * entropy

        return loss

    def clear(self):
        self.log_probs.clear()
        self.state_values.clear()
        self.rewards.clear()


def train(render=False, gamma=0.99, lr=0.02, betas=(0.9, 0.999), entropy_weight=0.01, max_episodes=1000, print_freq=10, save_threshold=2500):
    env = gym.make("LunarLander-v2")

    model = ACModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

    running_reward = 0
    for i in range(max_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        for t in range(10000):
            action = model(state)
            state, reward, terminated, truncated, _ = env.step(action)
            model.rewards.append(reward)
            running_reward += reward
            if render and i > 100:
                env.render()
            if terminated or truncated:
                break

        optimizer.zero_grad()
        loss = model.loss(model.rewards, gamma, entropy_weight)
        loss.backward()
        optimizer.step()
        model.clear()

        if running_reward > save_threshold:
            torch.save(model.state_dict(), 'policy{}.pth'.format(lr))
            print("Done! saved policy{}".format(lr))
            break

        if i % print_freq == 0:
            running_reward = running_reward / print_freq
            print('Episode {}\tmean reward: {:.2f}'.format(i + 1, running_reward))
            running_reward = 0


if __name__ == '__main__':
    train(render=True, gamma=0.99, lr=0.02, betas=(0.9, 0.999), entropy_weight=0.01, max_episodes=1000, print_freq=10, save_threshold=2500)

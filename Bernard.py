import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np
import os
import datetime
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


class ACModel(nn.Module):
    def __init__(self):
        super(ACModel, self).__init__()
        self.actor_common = nn.Linear(8, 128)

        self.actor_action = nn.Linear(128, 4)
        self.critic_value = nn.Linear(128, 1)

        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.rewards_log = []

    def forward(self, state):

        state = torch.from_numpy(state).float()
        state = F.relu(self.actor_common(state))
        state_value = self.critic_value(state)

        action_probs = F.softmax(self.actor_action(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.log_probs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculate_entropy(self, prob_distribution):
        dist = Categorical(prob_distribution)
        entropy = dist.entropy().mean()
        return entropy


    def loss(self, gamma=0.99, entropy_weight=0.01, use_baseline=False):
        #discount the rewards
        rewards = []
        discounted_reward = 0
        for reward in self.rewards[::-1]:
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = np.array(rewards)
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        rewards = torch.tensor(rewards)


        # Calculate loss using advantage
        loss = 0
        for log_prob, value, reward in zip(self.log_probs, self.state_values, rewards):
            if use_baseline:
                advantage = reward - value.item()
            else:
                advantage = reward
            loss += (-log_prob * advantage) + F.smooth_l1_loss(value, reward)

        # Add entropy term
        entropy = self.calculate_entropy(torch.stack(self.log_probs))
        loss = loss - (entropy_weight * entropy)

        return loss

    def clear(self):
        self.log_probs.clear()
        self.state_values.clear()
        self.rewards.clear()



def reinforce(policy, optimizer, gamma):
    R = 0
    policy_loss = []
    rewards = []
    eps = np.finfo(np.float32).eps.item()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.log_probs[:]


def train(render=False, gamma=0.99, lr=0.02, betas=(0.9, 0.999),
          entropy_weight=0.01, n_steps=5, num_episodes=1000,
          max_steps=10000, print_interval=10, method="a2c", use_baseline=True,
          N_bootstrap=5):  # Update frequency: Perform optimization step every X steps
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
        env.metadata['render_fps'] = 2048
    else:
        env = gym.make("LunarLander-v2")

    model = ACModel()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    running_reward = 0
    for i in range(num_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        episode_rewards = []
        for t in range(max_steps):
            action = model(state)
            state, reward, terminated, truncated, _ = env.step(action)
            model.rewards.append(reward)
            episode_rewards.append(reward)
            running_reward += reward

            if t % N_bootstrap == 0 or terminated or truncated:  # Perform optimization step every X steps
                optimizer.zero_grad()
                loss = model.loss(gamma, entropy_weight=entropy_weight, use_baseline=use_baseline)
                loss.backward()
                optimizer.step()
                model.clear()

            if terminated or truncated:
                break

        if i % print_interval == 0:
            print('Episode {}\t mean reward: {:.2f}'.format(i + 1, np.mean(model.rewards_log[-print_interval:])))

        model.rewards_log.append(sum(episode_rewards))



    # Save data
    directories = ['Data/rewards', 'Data/models']
    for directory in directories:
      if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")

    with open(os.path.join('Data/rewards', 'Rewards_gamma={}_lr={}_betas={}_entropy={}_nsteps={}_numepisodes={}_methods={}_usebaseline={}_{}.csv'
                                    .format(gamma, lr, betas, entropy_weight, n_steps, num_episodes, method, use_baseline, datetime.datetime.now().strftime("%d-%m_%H_%M"))), 'w') as file:
        for reward in model.rewards_log:
            file.write(str(reward) + '\n')
    # Save model data
    torch.save(model.state_dict(), os.path.join('Data/models',  'Model_gamma={}_lr={}_betas={}_entropy={}_nsteps={}_numepisodes={}_methods={}_usebaseline={}_{}.csv'
                                    .format(gamma, lr, betas, entropy_weight, n_steps, num_episodes, method, use_baseline, datetime.datetime.now().strftime("%d-%m_%H_%M"))))

    print('Done! Saved data to "{}" folder.'.format('Data'))
    return model.rewards_log


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    rewards = train(render=True,
              gamma=0.999,
              lr=0.01,
              betas=(0.9, 0.999),
              entropy_weight=0.1,
              num_episodes=500,
              max_steps=10000,
              print_interval=10,
              method="a2c",
              use_baseline=True,
              N_bootstrap=200)

    plt.plot(savgol_filter(rewards, 25, 3))

if __name__ == '__main__':
    main()
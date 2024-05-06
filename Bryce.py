
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import datetime



class REINFORCEModel(nn.Module):
    def __init__(self):
        super(REINFORCEModel, self).__init__()
        self.actor_common = nn.Linear(8, 128)

        self.actor_action = nn.Linear(128, 4)
        #self.critic_value = nn.Linear(128, 1)

        self.log_probs = []
        #self.state_values = []
        self.rewards = []

    def forward(self, state):

        state = torch.from_numpy(state).float()
        state = F.relu(self.actor_common(state))

        #state_value = self.critic_value(state)

        action_probs = F.softmax(self.actor_action(state), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.log_probs.append(action_distribution.log_prob(action))
        #self.state_values.append(state_value)

        return action.item()

    def calculate_entropy(self, prob_distribution):
        dist = Categorical(prob_distribution)
        entropy = dist.entropy().mean()
        return entropy

    def calculate_n_step_returns(self, n=5, gamma=0.99):
        n_step_returns = []
        rewards = (self.rewards - np.mean(self.rewards)) / (np.std(self.rewards))
        for i in reversed(range(len(rewards) - n + 1)):
            n_step_return = sum([rewards[i + j] * (gamma ** j) for j in range(n)])
            n_step_returns.insert(0, n_step_return)  # Prepend to the beginning
        return n_step_returns

    def loss(self, gamma=0.99, entropy_weight=0.01, n_steps=5):
        '''
        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        for reward in self.rewards[::-1]:
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()
        '''
        n_step_returns = self.calculate_n_step_returns(n=n_steps, gamma=gamma)
        n_step_returns = torch.tensor(n_step_returns)

        # Calculate loss using advantage
        loss = 0
        for log_prob, value, n_step_return in zip(self.log_probs, self.state_values, n_step_returns):
            advantage = n_step_return - value.item()
            loss += (-log_prob * advantage) + F.smooth_l1_loss(value, n_step_return)

        # Add entropy term
        entropy = self.calculate_entropy(torch.stack(self.log_probs))
        loss -= entropy_weight * entropy

        return loss

    def clear(self):
        self.log_probs.clear()
        self.state_values.clear()
        self.rewards.clear()



def reinforce(policy, optimizer, gamma):
    R = 0
    policy_loss = []
    rewards = []
    policy_rewards = []
    for r in policy.rewards[::-1]:
        policy_rewards.append(r)
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std())
    for log_prob, reward in zip(policy.log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    # print("policy_loss",policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.log_probs[:]
    return sum(policy_rewards)

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


def train(render = False, gamma=0.99, lr=0.02, betas=(0.9, 0.999),
          entropy_weight=0.01, num_episodes=1000,
          max_steps=10000, print_interval=10, method = "sac", N_bootstrap = 200,
          window_length = 5, use_baseline = True, save = False):
    """Note method can either be 'sac' or 'reinforce'"""
    if method == "reinforce":
        if render:
            env = gym.make("LunarLander-v2", render_mode="human")
            env.metadata['render_fps'] = 2048
        else:
            env = gym.make("LunarLander-v2")

        model = ACModel()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        rewards_lst = []
        running_reward = 0
        export_reward = np.zeros(int(num_episodes/print_interval))
        for i in range(0, num_episodes):
            state, _ = env.reset()
            terminated = truncated = False
            for t in range(max_steps):
                action = model(state)
                state, reward, terminated, truncated, _ = env.step(action)
                model.rewards.append(reward)
                running_reward += reward

                if terminated or truncated:
                    break
            rewards_ = reinforce(model, optimizer, gamma)
            rewards_lst.append(rewards_)

          # plt.plot(rewards_lst, color = "k", alpha = .3)
          # plt.plot(savgol_filter(rewards_lst, 25, 1), color = "blue")
          # plt.title(method)
        if save:
          gammastr = str(gamma).replace(".","__")
          lrstr = str(lr).replace(".","__")
          betasstr = str(betas).replace(".","__").replace(",","--")
          entropy_weightstr = str(entropy_weight).replace(".","__")
          num_episodesstr = str(num_episodes).replace(".","__")
          methodstr = str(method).replace(".","__")
          use_baselinestr = str(use_baseline).replace(".","__")
          DateTime = str(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")).replace(".","__")
          with open(str('/home/Rewards_gamma-'+gammastr+'_lr-'+lrstr+
                    '_betas-'+betasstr+'_entropy-'+entropy_weightstr+
                    '_numepisodes-'+num_episodesstr+
                    '_method-'+methodstr+'_usebaseline-'+use_baselinestr+
                    '_'+DateTime+'.csv'), 'w') as file:
              for reward in rewards_lst:
                  file.write(str(reward) + '\n')
          return rewards_lst, method

        if i % print_interval == 0:
            print('Episode {}\t mean reward: {:.2f}'.format(i + 1, np.mean(model.rewards_log[-print_interval:])))



    elif method == "a2c":
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
        # plt.plot(model.rewards_log, color = "k", alpha = .3)
        # plt.plot(savgol_filter(model.rewards_log, 25, 1), color = "blue")
        # plt.title(method)
        if save:
          gammastr = str(gamma).replace(".","__")
          lrstr = str(lr).replace(".","__")
          betasstr = str(betas).replace(".","__").replace(",","--")
          entropy_weightstr = str(entropy_weight).replace(".","__")
          num_episodesstr = str(num_episodes).replace(".","__")
          methodstr = str(method).replace(".","__")
          use_baselinestr = str(use_baseline).replace(".","__")
          DateTime = str(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")).replace(".","__")
          with open(str('Rewards_gamma-'+gammastr+'_lr-'+lrstr+
                    '_betas-'+betasstr+'_entropy-'+entropy_weightstr+
                    '_numepisodes-'+num_episodesstr+
                    '_method-'+methodstr+'_usebaseline-'+use_baselinestr+
                    '_'+DateTime+'.csv'), 'w') as file:
              for reward in model.rewards_log:
                  file.write(str(reward) + '\n')

        return model.rewards_log, method





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    plt.figure()
    rewards, method = train(render=False,
                            gamma=.99,
                            lr=0.0075,
                            betas=(0.9, 0.999),
                            entropy_weight=.1,
                            num_episodes=10,
                            max_steps=10000,
                            print_interval=100,
                            method="reinforce",
                            use_baseline=False,
                            N_bootstrap=1,
                            window_length = 75,
                            save = True)
    plt.plot(rewards, alpha = .1, color = "k")
    plt.plot(savgol_filter(rewards, 5, 1))
    plt.title(method)


if __name__ == '__main__':
    main()

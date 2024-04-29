from Bernard import train
import datetime

n_steps = [1, 5, 20, 50, 100]
rewards = []

for n_steps in n_steps:
    rewards.append(train(render=True,
      gamma=0.99,
      lr=0.01,
      betas=(0.9, 0.999),
      entropy_weight=0.01,
      n_steps=10,
      num_episodes=300,
      max_steps=10000,
      print_interval=n_steps,
      method="a2c",
      use_baseline=False))


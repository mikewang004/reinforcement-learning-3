from Bernard import train
import datetime

print(datetime.datetime.now().strftime("%d-%m_%H:%M"))
'''
rewards = []

for n_steps in (1, 10, 100, 1000, 10000):
    train(render=True,
      gamma=0.99,
      lr=0.01,
      betas=(0.9, 0.999),
      entropy_weight=0.01,
      n_steps=10,
      num_episodes=300,
      max_steps=10000,
      print_interval=n_steps,
      method="a2c",
      use_baseline=False)
'''
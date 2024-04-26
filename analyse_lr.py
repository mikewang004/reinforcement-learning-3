import numpy as np
import matplotlib.pyplot as plt


lr_data = np.loadtxt("data/reinforce-lr-test.txt")
lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
x = np.arange(0, 100)

for i in range(len(lr_data)):
    print(lr_data)
    plt.plot(x*10, lr_data[i], label = lr_list[i])
    plt.legend()

plt.title("Learning rate tuning for REINFORCE")
plt.xlabel("episode number")
plt.ylabel("reward")
plt.show()
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

x = []

with open('Data/rewards/Rewards_gamma=0.99_lr=0.01_betas=(0.9, 0.999)_entropy=0.1_nsteps=5_numepisodes=500_methods=a2c_usebaseline=True_01-05_11_42.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip header if present
    for row in reader:
        x.append(float(row[0]))  # assuming first column is x


plt.plot(savgol_filter(x, 25, 3))
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Your Title Here')
plt.show()
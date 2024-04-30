import csv
import matplotlib.pyplot as plt

x = []

with open('Data/rewards/Rewards_gamma=0.999_lr=0.01_betas=(0.99, 0.999)_entropy=0.01_nsteps=5_numepisodes=500_methods=a2c_usebaseline=True_30-04_17_24.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip header if present
    for row in reader:
        x.append(float(row[0]))  # assuming first column is x


plt.plot(x)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Your Title Here')
plt.show()
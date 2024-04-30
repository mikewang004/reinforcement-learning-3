from matplotlib import pyplot as plt
import os
import numpy as np
import csv

from matplotlib import pyplot as plt
import os
import numpy as np
import csv

# Directory containing the CSV files
directory = 'data/rewards'

# List to store the mean values from all CSV files
mean_values = []

# Iterate through each CSV file, read the data, calculate the mean, and store it
for file in os.listdir(directory):
    if file.endswith(".csv"):
        file_path = os.path.join(directory, file)

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data = np.array(list(reader)[1:]).astype(np.cfloat)  # Skip the header row and convert data to float
            mean_value = np.mean(data)  # Calculate the mean of all values in the file
            mean_values.append(mean_value)

# Plot the mean values
plt.plot(mean_values)
plt.xlabel('File Index')
plt.ylabel('Mean Value')
plt.title('Mean Value from CSV Files')
plt.show()
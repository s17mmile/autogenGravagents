import numpy as np
import matplotlib.pyplot as plt

# Load data from the text files
h1_data = np.loadtxt('gwosc_gw150914_h1.txt', comments='#')
l1_data = np.loadtxt('gwosc_gw150914_l1.txt', comments='#')

# Extract strain values and time
h1_strain = h1_data[:, 0]
# Assuming the time starts from 0 and increments by 1 for each data point
h1_time = np.arange(len(h1_strain))

l1_strain = l1_data[:, 0]
l1_time = np.arange(len(l1_strain))

# Plot H1 strain
plt.figure(figsize=(10, 5))
plt.plot(h1_time, h1_strain, label='H1 Strain', color='blue')
plt.title('H1 Strain vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.grid()
plt.savefig('H1_strain.png')

# Plot L1 strain
plt.figure(figsize=(10, 5))
plt.plot(l1_time, l1_strain, label='L1 Strain', color='orange')
plt.title('L1 Strain vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.grid()
plt.savefig('L1_strain.png')
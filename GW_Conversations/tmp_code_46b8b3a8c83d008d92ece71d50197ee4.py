import numpy as np
import matplotlib.pyplot as plt

# Load data from the text files
try:
    h1_data = np.loadtxt('gwosc_gw150914_h1.txt', comments='#')
    l1_data = np.loadtxt('gwosc_gw150914_l1.txt', comments='#')
except Exception as e:
    print(f'Error loading data: {e}')
    exit()

# Extract strain values and time
h1_strain = h1_data
h1_time = np.arange(len(h1_strain))  # Adjust if actual time intervals are known

l1_strain = l1_data
l1_time = np.arange(len(l1_strain))  # Adjust if actual time intervals are known

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
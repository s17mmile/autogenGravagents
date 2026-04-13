import numpy as np
import matplotlib.pyplot as plt

# Constants
sampling_rate = 16384  # samples per second

def plot_strain(file_name, color, title, output_file):
    try:
        data = np.loadtxt(file_name, skiprows=3)
        if data.size == 0:
            raise ValueError('Data is empty')
        time = np.arange(len(data)) / sampling_rate
        plt.figure(figsize=(10, 5))
        plt.plot(time, data, color=color)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.grid()
        plt.savefig(output_file)
        plt.close()
        print(f'{output_file} saved successfully.')  # Confirmation message
    except Exception as e:
        print(f'Error processing {file_name}: {e}')  # Error handling

# Plot H1 strain
plot_strain('gwosc_gw150914_h1.txt', 'blue', 'H1 Strain vs Time', 'H1_strain.png')

# Plot L1 strain
plot_strain('gwosc_gw150914_l1.txt', 'red', 'L1 Strain vs Time', 'L1_strain.png')
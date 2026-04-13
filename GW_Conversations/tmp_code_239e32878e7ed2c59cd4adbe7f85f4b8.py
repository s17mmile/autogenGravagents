import numpy as np
import matplotlib.pyplot as plt

# Constants
sampling_rate = 16384  # samples per second

def whiten_signal(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std  # Simple whitening process


def plot_whitened_signal(time, whitened_data, color, title, output_file):
    plt.figure(figsize=(10, 5))
    plt.plot(time, whitened_data, color=color)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Whitened Strain')
    plt.grid()
    plt.savefig(output_file)
    plt.close()
    print(f'{output_file} saved successfully.')  # Confirmation message

try:
    # Load and whiten H1 data
    h1_data = np.loadtxt('gwosc_gw150914_h1.txt', skiprows=3)
    if h1_data.size == 0:
        raise ValueError('H1 data is empty')
    whitened_h1 = whiten_signal(h1_data)
    h1_time = np.arange(len(whitened_h1)) / sampling_rate
    plot_whitened_signal(h1_time, whitened_h1, 'blue', 'Whitened H1 Strain vs Time', 'H1_strain_whitened.png')

    # Load and whiten L1 data
    l1_data = np.loadtxt('gwosc_gw150914_l1.txt', skiprows=3)
    if l1_data.size == 0:
        raise ValueError('L1 data is empty')
    whitened_l1 = whiten_signal(l1_data)
    l1_time = np.arange(len(whitened_l1)) / sampling_rate
    plot_whitened_signal(l1_time, whitened_l1, 'red', 'Whitened L1 Strain vs Time', 'L1_strain_whitened.png')

except Exception as e:
    print(f'Error: {e}')  # Error handling
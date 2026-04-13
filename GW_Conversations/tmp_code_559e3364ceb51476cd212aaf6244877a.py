import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Constants
sampling_rate = 16384  # samples per second
lowcut = 20.0  # Low cutoff frequency for bandpass filter
highcut = 400.0  # High cutoff frequency for bandpass filter

# Bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply bandpass filter to data

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

try:
    # Load and filter H1 data
    h1_data = np.loadtxt('gwosc_gw150914_h1.txt', skiprows=3)
    if h1_data.size == 0:
        raise ValueError('H1 data is empty')
    whitened_h1 = bandpass_filter(h1_data, lowcut, highcut, sampling_rate)

    # Load and filter L1 data
    l1_data = np.loadtxt('gwosc_gw150914_l1.txt', skiprows=3)
    if l1_data.size == 0:
        raise ValueError('L1 data is empty')
    whitened_l1 = bandpass_filter(l1_data, lowcut, highcut, sampling_rate)

    # Time arrays
    h1_time = np.arange(len(whitened_h1)) / sampling_rate
    l1_time = np.arange(len(whitened_l1)) / sampling_rate

    # Plot whitened H1 strain
    plt.figure(figsize=(10, 5))
    plt.plot(h1_time, whitened_h1, color='blue')
    plt.title('Whitened H1 Strain vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Whitened Strain')
    plt.grid()
    plt.savefig('H1_strain_whitened.png')
    print('H1_strain_whitened.png saved successfully.')  # Confirmation message

    # Plot whitened L1 strain
    plt.figure(figsize=(10, 5))
    plt.plot(l1_time, whitened_l1, color='red')
    plt.title('Whitened L1 Strain vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Whitened Strain')
    plt.grid()
    plt.savefig('L1_strain_whitened.png')
    print('L1_strain_whitened.png saved successfully.')  # Confirmation message

except Exception as e:
    print(f'Error: {e}')  # Error handling
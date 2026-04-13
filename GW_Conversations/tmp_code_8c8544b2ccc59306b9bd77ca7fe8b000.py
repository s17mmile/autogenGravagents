import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform
from pycbc.filter import highpass, lowpass
from pycbc.types import TimeSeries

# Constants
sampling_rate = 16384  # samples per second
lowcut = 20.0  # Low cutoff frequency for bandpass filter
highcut = 400.0  # High cutoff frequency for bandpass filter

# Load and whiten H1 data
h1_data = np.loadtxt('gwosc_gw150914_h1.txt', skiprows=3)

# Create TimeSeries object
h1_ts = TimeSeries(h1_data, delta_t=1/sampling_rate)

# Apply whitening using PyCBC
h1_whitened = h1_ts.whiten(lowcut, highcut)

# Load and whiten L1 data
l1_data = np.loadtxt('gwosc_gw150914_l1.txt', skiprows=3)

# Create TimeSeries object
l1_ts = TimeSeries(l1_data, delta_t=1/sampling_rate)

# Apply whitening using PyCBC
l1_whitened = l1_ts.whiten(lowcut, highcut)

# Time arrays
h1_time = np.arange(len(h1_whitened)) / sampling_rate
l1_time = np.arange(len(l1_whitened)) / sampling_rate

# Plot whitened H1 strain
plt.figure(figsize=(10, 5))
plt.plot(h1_time, h1_whitened, color='blue')
plt.title('Whitened H1 Strain vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Whitened Strain')
plt.grid()
plt.savefig('H1_strain_whitened.png')

# Plot whitened L1 strain
plt.figure(figsize=(10, 5))
plt.plot(l1_time, l1_whitened, color='red')
plt.title('Whitened L1 Strain vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Whitened Strain')
plt.grid()
plt.savefig('L1_strain_whitened.png')
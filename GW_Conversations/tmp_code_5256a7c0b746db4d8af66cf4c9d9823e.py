import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform, psd, filter, timeseries
from pycbc.detector import Detector
from pycbc.psd import aLIGO

# Load strain data from text files with error handling
try:
    h1_data = np.loadtxt('gwosc_gw150914_h1.txt', comments='#')
    l1_data = np.loadtxt('gwosc_gw150914_l1.txt', comments='#')
except Exception as e:
    print('Error loading data:', e)
    exit()

# Validate data
if h1_data.size == 0 or l1_data.size == 0:
    print('Error: One of the data arrays is empty.')
    exit()

# Create time arrays
sampling_rate = 16384  # Hz
gps_start = 1126259447
duration = 32

time = np.arange(0, duration, 1/sampling_rate)

# Plot strain vs time for both detectors
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, h1_data)
plt.title('Strain Data for H1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, l1_data)
plt.title('Strain Data for L1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid()
plt.tight_layout()
plt.savefig('strain_data.png')

# Whiten the strain data
h1_strain = timeseries.TimeSeries(h1_data, delta_t=1/sampling_rate)
l1_strain = timeseries.TimeSeries(l1_data, delta_t=1/sampling_rate)

h1_whitened = h1_strain.whiten()
l1_whitened = l1_strain.whiten()

# Plot whitened strain data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(h1_whitened.sample_times, h1_whitened)
plt.title('Whitened Strain Data for H1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Whitened Strain')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(l1_whitened.sample_times, l1_whitened)
plt.title('Whitened Strain Data for L1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Whitened Strain')
plt.grid()
plt.tight_layout()
plt.savefig('whitened_strain_data.png')

# Apply band-pass filter
h1_filtered = h1_whitened.bandpass(30, 250)
l1_filtered = l1_whitened.bandpass(30, 250)

# Plot filtered data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(h1_filtered.sample_times, h1_filtered)
plt.title('Filtered Strain Data for H1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Filtered Strain')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(l1_filtered.sample_times, l1_filtered)
plt.title('Filtered Strain Data for L1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Filtered Strain')
plt.grid()
plt.tight_layout()
plt.savefig('filtered_strain_data.png')

# Generate and plot Power Spectral Density (PSD)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.psd(h1_strain, NFFT=2048, Fs=sampling_rate, label='H1 Original')
plt.psd(h1_whitened, NFFT=2048, Fs=sampling_rate, label='H1 Whitened')
plt.title('PSD for H1 Detector')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.psd(l1_strain, NFFT=2048, Fs=sampling_rate, label='L1 Original')
plt.psd(l1_whitened, NFFT=2048, Fs=sampling_rate, label='L1 Whitened')
plt.title('PSD for L1 Detector')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('psd_data.png')

# Generate Q-transform spectrograms
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.specgram(h1_filtered, NFFT=2048, Fs=sampling_rate, noverlap=1024)
plt.title('Q-Transform Spectrogram for H1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid()

plt.subplot(2, 1, 2)
plt.specgram(l1_filtered, NFFT=2048, Fs=sampling_rate, noverlap=1024)
plt.title('Q-Transform Spectrogram for L1 Detector')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.tight_layout()
plt.savefig('spectrogram_data.png')

# Summary of key characteristics
print('Summary of GW150914 Event:')
print('The event occurred on GPS time:', gps_start)
print('Duration of the event:', duration, 'seconds')
print('Key frequency range of interest: 30-250 Hz')

import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform, psd, filter, detector, fetch
from pycbc.events import event
from pycbc.gravitational_wave import strain
from pycbc.psd import aLIGO
from pycbc.qtransform import qtransform

# Constants for analysis
start_time = 1126259462.4  # GPS time for GW150914
end_time = start_time + 4  # 4 seconds duration
segment_length = 4  # Segment length for PSD estimation
q_transform_window = 1  # Window length for q-transform
q_transform_overlap = 0.1  # Overlap for q-transform

try:
    # Fetch strain data for GW150914
    H1 = fetch.strain('H1', start_time, end_time)
    L1 = fetch.strain('L1', start_time, end_time)
except Exception as e:
    print(f'Error fetching strain data: {e}')
    raise

# Plot strain vs time for both detectors
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(H1.sample_times, H1.data, label='H1 Strain')
plt.title('GW150914 Strain Data - H1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Strain')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(L1.sample_times, L1.data, label='L1 Strain', color='orange')
plt.title('GW150914 Strain Data - L1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Strain')
plt.legend()
plt.tight_layout()
plt.savefig('strain_data.png')
plt.show()

# Whiten the strain data
H1_psd = psd.estimate(H1, segment_length)
L1_psd = psd.estimate(L1, segment_length)
H1_whitened = filter.whiten(H1, H1_psd)
L1_whitened = filter.whiten(L1, L1_psd)

# Plot whitened strain data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(H1_whitened.sample_times, H1_whitened.data, label='H1 Whitened Strain')
plt.title('Whitened Strain Data - H1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Whitened Strain')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(L1_whitened.sample_times, L1_whitened.data, label='L1 Whitened Strain', color='orange')
plt.title('Whitened Strain Data - L1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Whitened Strain')
plt.legend()
plt.tight_layout()
plt.savefig('whitened_strain_data.png')
plt.show()

# Apply band-pass filter
H1_filtered = filter.bandpass(H1_whitened, 30, 250)
L1_filtered = filter.bandpass(L1_whitened, 30, 250)

# Plot filtered data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(H1_filtered.sample_times, H1_filtered.data, label='H1 Filtered Strain')
plt.title('Filtered Strain Data - H1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Filtered Strain')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(L1_filtered.sample_times, L1_filtered.data, label='L1 Filtered Strain', color='orange')
plt.title('Filtered Strain Data - L1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Filtered Strain')
plt.legend()
plt.tight_layout()
plt.savefig('filtered_strain_data.png')
plt.show()

# Generate and plot power spectral density (PSD)
H1_psd = psd.estimate(H1, segment_length)
L1_psd = psd.estimate(L1, segment_length)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.loglog(H1_psd.sample_frequencies, H1_psd.data, label='H1 PSD')
plt.title('Power Spectral Density - H1')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()

plt.subplot(2, 1, 2)
plt.loglog(L1_psd.sample_frequencies, L1_psd.data, label='L1 PSD', color='orange')
plt.title('Power Spectral Density - L1')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.tight_layout()
plt.savefig('psd_data.png')
plt.show()

# Generate q-transform spectrograms
H1_spectrogram = qtransform(H1_filtered, q_transform_window, q_transform_overlap)
L1_spectrogram = qtransform(L1_filtered, q_transform_window, q_transform_overlap)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.imshow(H1_spectrogram, aspect='auto', origin='lower', extent=[0, 4, 30, 250], cmap='inferno')
plt.title('Q-Transform Spectrogram - H1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Frequency (Hz)')

plt.subplot(2, 1, 2)
plt.imshow(L1_spectrogram, aspect='auto', origin='lower', extent=[0, 4, 30, 250], cmap='inferno')
plt.title('Q-Transform Spectrogram - L1')
plt.xlabel('Time (GPS seconds)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('spectrogram_data.png')
plt.show()

# Placeholder for summarizing key characteristics
# TODO: Implement summarization of key characteristics based on analysis results.
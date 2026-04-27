# filename: gw150914_analysis_final.py

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy import datasets
from pycbc import waveform
from pycbc.psd import aLIGO
from pycbc.filter import matched_filter

# Function to save plots with tight layout
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Task 1: Data fetching
# Define the event and file paths
event_id = 'GW150914'
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
file_h1 = os.path.join(parent_dir, 'gwosc_gw150914_h1.hdf5')
file_l1 = os.path.join(parent_dir, 'gwosc_gw150914_l1.hdf5')

# Check if files exist
if os.path.exists(file_h1) and os.path.exists(file_l1):
    print('Loading data from disk...')
    h1_data = TimeSeries.read(file_h1)
    l1_data = TimeSeries.read(file_l1)
else:
    print('Fetching data from GWOSC...')
    gps_time = datasets.get_event_time(event_id)
    start_time = gps_time - 8  # 8 seconds before merger
    end_time = gps_time + 4    # 4 seconds after merger
    h1_data = datasets.get_data('H1', start_time, end_time)
    l1_data = datasets.get_data('L1', start_time, end_time)
    # Save to disk
    h1_data.write(file_h1)
    l1_data.write(file_l1)

# Plot strain data
plt.figure(figsize=(12, 6))
plt.plot(h1_data.times, h1_data, label='H1 Strain')
plt.plot(l1_data.times, l1_data, label='L1 Strain')
plt.title('Strain Data for GW150914')
plt.xlabel('GPS Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.grid()
save_plot('strain_data_gw150914_task1.png')

# Task 2: Data filtering
# Whiten the signals
h1_whitened = h1_data.whiten()
l1_whitened = l1_data.whiten()

# Plot whitened data
plt.figure(figsize=(12, 6))
plt.plot(h1_whitened.times, h1_whitened, label='H1 Whitened')
plt.plot(l1_whitened.times, l1_whitened, label='L1 Whitened')
plt.title('Whitened Strain Data')
plt.xlabel('GPS Time (s)')
plt.ylabel('Whitened Strain')
plt.legend()
plt.grid()
save_plot('whitened_strain_data_task2.png')

# Apply band-pass filter
h1_filtered = h1_whitened.bandpass(30, 250)
l1_filtered = l1_whitened.bandpass(30, 250)

# Plot filtered data
plt.figure(figsize=(12, 6))
plt.plot(h1_filtered.times, h1_filtered, label='H1 Filtered')
plt.plot(l1_filtered.times, l1_filtered, label='L1 Filtered')
plt.title('Filtered Strain Data')
plt.xlabel('GPS Time (s)')
plt.ylabel('Filtered Strain')
plt.legend()
plt.grid()
save_plot('filtered_strain_data_task2.png')

# Task 3: Q-Transform
plt.figure(figsize=(12, 6))
plt.specgram(h1_filtered, NFFT=1024, Fs=1.0, Fc=0, noverlap=512, cmap='inferno')
plt.colorbar(label='Normalized Energy')
plt.clim(0, 25)
plt.title('Q-Transform for H1')
plt.xlabel('GPS Time (s)')
plt.ylabel('Frequency (Hz)')
save_plot('q_transform_h1_task3.png')

plt.figure(figsize=(12, 6))
plt.specgram(l1_filtered, NFFT=1024, Fs=1.0, Fc=0, noverlap=512, cmap='inferno')
plt.colorbar(label='Normalized Energy')
plt.clim(0, 25)
plt.title('Q-Transform for L1')
plt.xlabel('GPS Time (s)')
plt.ylabel('Frequency (Hz)')
save_plot('q_transform_l1_task3.png')

# Task 4: Data format conversion
# Convert to PyCBC TimeSeries
h1_pycbc = h1_data.to_pycbc()
l1_pycbc = l1_data.to_pycbc()

# Ensure consistent sample rates and lengths
if h1_pycbc.sample_rate != l1_pycbc.sample_rate:
    l1_pycbc = l1_pycbc.resample(h1_pycbc.sample_rate)
if len(h1_pycbc) != len(l1_pycbc):
    min_length = min(len(h1_pycbc), len(l1_pycbc))
    h1_pycbc = h1_pycbc[:min_length]
    l1_pycbc = l1_pycbc[:min_length]

# Task 5: PyCBC template creation
masses = [10, 20, 30, 40]
for mass in masses:
    template = waveform.get_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, spin1z=0, spin2z=0)
    if len(template) > 0.2 * h1_pycbc.duration:
        template = template.crop(0, h1_pycbc.duration)
        # Scale template
        template *= (h1_pycbc.max() / template.max())
        # Plot overlay
        plt.figure(figsize=(12, 6))
        plt.plot(h1_pycbc.times, h1_pycbc, label='H1 Strain')
        plt.plot(template.times, template, label='Template', alpha=0.7)
        plt.title(f'Template for {mass} Solar Masses')
        plt.xlabel('GPS Time (s)')
        plt.ylabel('Strain')
        plt.legend()
        plt.grid()
save_plot(f'template_{mass}_task5.png')

# Task 6: Calculating the Power Spectral Density
h1_psd = aLIGO(h1_pycbc)
l1_psd = aLIGO(l1_pycbc)

# Interpolate and truncate PSD
h1_psd = h1_psd.interpolate(30)
l1_psd = l1_psd.interpolate(30)

# Plot PSDs
plt.figure(figsize=(12, 6))
plt.loglog(h1_psd.frequencies, h1_psd, label='H1 PSD')
plt.loglog(l1_psd.frequencies, l1_psd, label='L1 PSD')
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.grid()
save_plot('psd_plot_task6.png')

# Task 7: Matched filtering
best_fit = {}
for mass in masses:
    template = waveform.get_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, spin1z=0, spin2z=0)
    h1_snr = matched_filter(template, h1_pycbc)
    l1_snr = matched_filter(template, l1_pycbc)
    # Plot SNR time series
    plt.figure(figsize=(12, 6))
    plt.plot(h1_snr.sample_times, h1_snr, label='H1 SNR')
    plt.plot(l1_snr.sample_times, l1_snr, label='L1 SNR')
    plt.title(f'Matched Filtering SNR for {mass} Solar Masses')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('SNR')
    plt.legend()
    plt.grid()
save_plot(f'snr_{mass}_task7.png')
    # Store best fit
    best_fit[mass] = (h1_snr.max(), l1_snr.max())

print('Best fit templates:', best_fit)
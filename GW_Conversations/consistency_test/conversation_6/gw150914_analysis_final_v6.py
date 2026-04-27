# filename: gw150914_analysis_final_v6.py

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.transform import q_transform
from gwosc import datasets
from pycbc import waveform, psd, filter
from pycbc.types import TimeSeries as PyCBC_TimeSeries

# Function to fetch data

def fetch_data(event_name, start_offset, end_offset):
    try:
        gps_time = datasets.get_event_time(event_name)
        start_time = gps_time + start_offset
        end_time = gps_time + end_offset
        print(f'Fetching data from {start_time} to {end_time}...')
        h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time)
        l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time)
        return h1_data, l1_data
    except Exception as e:
        print(f'Error fetching data: {e}')
        return None, None

# Function to plot data

def plot_data(times, data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(times, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(filename)
    plt.show()  # Show plot for interactive environments
    plt.close()

# Task 1: Data fetching

event_name = 'GW150914'
start_offset = -8  # seconds before merger
end_offset = 4     # seconds after merger

h1_file = '../gwosc_gw150914_h1.hdf5'
l1_file = '../gwosc_gw150914_l1.hdf5'

if os.path.exists(h1_file) and os.path.exists(l1_file):
    print('Loading data from existing files...')
    h1_data = TimeSeries.read(h1_file)
    l1_data = TimeSeries.read(l1_file)
else:
    h1_data, l1_data = fetch_data(event_name, start_offset, end_offset)
    if h1_data is not None and l1_data is not None:
        h1_data.write(h1_file)
        l1_data.write(l1_file)

# Plot strain data
plot_data(h1_data.times, h1_data, 'Strain Data for GW150914 (H1)', 'GPS Time (s)', 'Strain', 'strain_data_h1.png')
plot_data(l1_data.times, l1_data, 'Strain Data for GW150914 (L1)', 'GPS Time (s)', 'Strain', 'strain_data_l1.png')

# Task 2: Data filtering

# Whiten the signals
try:
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()
    plot_data(h1_whitened.times, h1_whitened, 'Whitened Strain Data (H1)', 'GPS Time (s)', 'Whitened Strain', 'whitened_strain_h1.png')
    plot_data(l1_whitened.times, l1_whitened, 'Whitened Strain Data (L1)', 'GPS Time (s)', 'Whitened Strain', 'whitened_strain_l1.png')
except Exception as e:
    print(f'Error during whitening: {e}')

# Apply band-pass filter
try:
    h1_filtered = h1_whitened.bandpass(30, 250)
    l1_filtered = l1_whitened.bandpass(30, 250)
    plot_data(h1_filtered.times, h1_filtered, 'Filtered Strain Data (H1)', 'GPS Time (s)', 'Filtered Strain', 'filtered_strain_h1.png')
    plot_data(l1_filtered.times, l1_filtered, 'Filtered Strain Data (L1)', 'GPS Time (s)', 'Filtered Strain', 'filtered_strain_l1.png')
except Exception as e:
    print(f'Error during filtering: {e}')

# Task 3: Q-Transform

# Using the correct Q-transform function from GWpy
try:
    dt = h1_filtered.delta_t  # Time step of the data
    q_transform_h1 = q_transform(h1_filtered, dt=dt, fmin=30, fmax=250)
    q_transform_l1 = q_transform(l1_filtered, dt=dt, fmin=30, fmax=250)
    plt.figure(figsize=(12, 6))
    plt.imshow(q_transform_h1, aspect='auto', origin='lower', extent=[0, h1_filtered.duration, 30, 250], cmap='inferno')
    plt.colorbar(label='Energy')
    plt.title('Q-Transform for H1')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig('q_transform_h1.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(q_transform_l1, aspect='auto', origin='lower', extent=[0, l1_filtered.duration, 30, 250], cmap='inferno')
    plt.colorbar(label='Energy')
    plt.title('Q-Transform for L1')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig('q_transform_l1.png')
    plt.show()
    plt.close()
except Exception as e:
    print(f'Error during Q-Transform: {e}')

# Task 4: Data format conversion

try:
    h1_pycbc = PyCBC_TimeSeries(h1_data, delta_t=h1_data.delta_t)
    l1_pycbc = PyCBC_TimeSeries(l1_data, delta_t=l1_data.delta_t)
    if h1_pycbc.sample_rate != l1_pycbc.sample_rate:
        l1_pycbc = l1_pycbc.resample(h1_pycbc.sample_rate)
except Exception as e:
    print(f'Error during data format conversion: {e}')

# Task 5: PyCBC template creation

masses = [10, 20, 30, 40]
approximant = 'IMRPhenomD'

for mass in masses:
    try:
        template = waveform.get_waveform(approximant=approximant, mass1=mass, mass2=mass, spin1z=0, spin2z=0)
        if len(template) > 0.2 * h1_pycbc.duration:
            template.resize(h1_pycbc.duration)
            template *= h1_pycbc.max() / template.max()
            plot_data(template.times, template, f'Template for {mass} Solar Masses (H1)', 'GPS Time (s)', 'Strain', f'template_{mass}_h1.png')
            np.save(f'template_{mass}.npy', template)
    except Exception as e:
        print(f'Error creating template for mass {mass}: {e}')

# Task 6: Calculating the Power Spectral Density

try:
    h1_psd = psd.estimate(h1_pycbc)
    l1_psd = psd.estimate(l1_pycbc)
    h1_psd = h1_psd.interpolate(30)
    l1_psd = l1_psd.interpolate(30)
    plot_data(h1_psd.frequencies, h1_psd, 'Power Spectral Density (H1)', 'Frequency (Hz)', 'PSD', 'psd_h1.png')
    plot_data(l1_psd.frequencies, l1_psd, 'Power Spectral Density (L1)', 'Frequency (Hz)', 'PSD', 'psd_l1.png')
except Exception as e:
    print(f'Error during PSD calculation: {e}')

# Task 7: Matched filtering

def matched_filtering(h1_pycbc, l1_pycbc, masses):
    best_fit_h1 = None
    best_fit_l1 = None
    max_snr_h1 = 0
    max_snr_l1 = 0

    for mass in masses:
        try:
            template = np.load(f'template_{mass}.npy')
            h1_snr = filter.match(h1_pycbc, template)
            l1_snr = filter.match(l1_pycbc, template)
            plt.figure(figsize=(12, 6))
            plt.plot(h1_snr.times, h1_snr, label='H1 SNR')
            plt.title(f'SNR for Template {mass} (H1)')
            plt.xlabel('GPS Time (s)')
            plt.ylabel('SNR')
            plt.legend()
            plt.grid()
            plt.savefig(f'snr_h1_{mass}.png')
            plt.show()
            plt.close()
            plt.figure(figsize=(12, 6))
            plt.plot(l1_snr.times, l1_snr, label='L1 SNR')
            plt.title(f'SNR for Template {mass} (L1)')
            plt.xlabel('GPS Time (s)')
            plt.ylabel('SNR')
            plt.legend()
            plt.grid()
            plt.savefig(f'snr_l1_{mass}.png')
            plt.show()
            plt.close()
            # Identify best fit
            if h1_snr.max() > max_snr_h1:
                max_snr_h1 = h1_snr.max()
                best_fit_h1 = mass
            if l1_snr.max() > max_snr_l1:
                max_snr_l1 = l1_snr.max()
                best_fit_l1 = mass
        except Exception as e:
            print(f'Error during matched filtering for mass {mass}: {e}')

    print(f'Best fit template for H1: {best_fit_h1} Solar Masses with SNR: {max_snr_h1}')
    print(f'Best fit template for L1: {best_fit_l1} Solar Masses with SNR: {max_snr_l1}')

matched_filtering(h1_pycbc, l1_pycbc, masses)
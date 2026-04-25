# filename: gw150914_analysis_final.py

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc import datasets
from pycbc import waveform, psd, filter as pycbc_filter

# Function to fetch data

def fetch_data(event, start_offset, end_offset):
    try:
        gps_time = datasets.get_event_time(event)
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
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Task 1: Data fetching

event = 'GW150914'
start_offset = -8  # seconds before merger
end_offset = 4     # seconds after merger

h1_file = '../gwosc_gw150914_h1.hdf5'
l1_file = '../gwosc_gw150914_l1.hdf5'

if os.path.exists(h1_file) and os.path.exists(l1_file):
    print('Loading data from disk...')
    h1_data = TimeSeries.read(h1_file)
    l1_data = TimeSeries.read(l1_file)
else:
    h1_data, l1_data = fetch_data(event, start_offset, end_offset)
    if h1_data is not None and l1_data is not None:
        h1_data.write(h1_file)
        l1_data.write(l1_file)

# Plot strain data
plot_data(h1_data.times, h1_data, 'Strain Data for GW150914', 'GPS Time (s)', 'Strain', 'strain_data.png')

# Task 2: Data filtering

# Whiten the data
h1_whitened = h1_data.whiten()
l1_whitened = l1_data.whiten()

# Plot whitened data
plot_data(h1_whitened.times, h1_whitened, 'Whitened Strain Data', 'GPS Time (s)', 'Whitened Strain', 'whitened_strain_data.png')

# Band-pass filter
h1_filtered = h1_whitened.bandpass(30, 250)
l1_filtered = l1_whitened.bandpass(30, 250)

# Plot filtered data
plot_data(h1_filtered.times, h1_filtered, 'Filtered Strain Data', 'GPS Time (s)', 'Filtered Strain', 'filtered_strain_data.png')

# Task 3: Q-Transform

if h1_filtered is not None and l1_filtered is not None:
    plt.figure(figsize=(12, 6))
    plt.specgram(h1_filtered.value, NFFT=1024, Fs=h1_filtered.sample_rate.value, noverlap=512)
    plt.colorbar(label='Normalized Energy')
    plt.clim(0, 25)
    plt.title('Q-Transform for H1')
    plt.tight_layout()
    plt.savefig('q_transform_h1.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.specgram(l1_filtered.value, NFFT=1024, Fs=l1_filtered.sample_rate.value, noverlap=512)
    plt.colorbar(label='Normalized Energy')
    plt.clim(0, 25)
    plt.title('Q-Transform for L1')
    plt.tight_layout()
    plt.savefig('q_transform_l1.png')
    plt.close()

# Task 4: Data format conversion

h1_pycbc = h1_data.to_pycbc()
l1_pycbc = l1_data.to_pycbc()

if h1_pycbc.sample_rate != l1_pycbc.sample_rate:
    l1_pycbc = l1_pycbc.resample(h1_pycbc.sample_rate)

# Task 5: PyCBC template creation

masses = [10, 20, 30, 40]
approximant = 'IMRPhenomD'

for mass in masses:
    template = waveform.get_fd_waveform(approximant=approximant, mass1=mass, mass2=mass, spin1z=0, spin2z=0)
    if template is not None and len(template) > 0.2 * h1_pycbc.duration:
        template = template.crop(0, h1_pycbc.duration)
        if template.max() != 0:
            template *= h1_pycbc.max() / template.max()
        plot_data(h1_pycbc.times, h1_pycbc, f'Template for Mass {mass} Solar Masses', 'GPS Time (s)', 'Strain', f'template_mass_{mass}.png')

# Task 6: Calculating the Power Spectral Density

h1_psd = psd.estimate(h1_pycbc, seglen=4)
l1_psd = psd.estimate(l1_pycbc, seglen=4)

h1_psd = h1_psd.interpolate(30)
l1_psd = l1_psd.interpolate(30)

# Plot PSD
plt.figure(figsize=(12, 6))
plt.loglog(h1_psd.frequencies, h1_psd, label='H1 PSD')
plt.loglog(l1_psd.frequencies, l1_psd, label='L1 PSD')
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('psd.png')
plt.close()

# Task 7: Matched filtering

snr_results = {}

for mass in masses:
    template = waveform.get_fd_waveform(approximant=approximant, mass1=mass, mass2=mass, spin1z=0, spin2z=0)
    if template is not None:
        h1_snr = pycbc_filter.match(h1_pycbc, template)
        l1_snr = pycbc_filter.match(l1_pycbc, template)
        snr_results[mass] = (h1_snr, l1_snr)
        plot_data(h1_snr.times, h1_snr, f'SNR for Template Mass {mass}', 'GPS Time (s)', 'SNR', f'snr_mass_{mass}.png')

# Report best fit templates
best_fit = {mass: max(snr_results[mass][0]) for mass in masses}
print('Best fit templates:', best_fit)
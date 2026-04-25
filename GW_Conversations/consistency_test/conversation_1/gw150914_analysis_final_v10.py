# filename: gw150914_analysis_final_v10.py

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc import datasets
from pycbc import waveform, psd, filter as pycbc_filter

# Constants
GPS_START = 1126059462  # GPS time for the merger
DURATION = 12  # seconds

# Helper function for plotting
def save_plot(x, y, title, xlabel, ylabel, filename, legend=None):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f'Plot saved: {filename}')  

# Task 1: Data fetching
try:
    # Check if data files exist
    h1_file = '../gwosc_gw150914_h1.hdf5'
    l1_file = '../gwosc_gw150914_l1.hdf5'
    if os.path.exists(h1_file) and os.path.exists(l1_file):
        print('Loading data from disk...')
        h1_data = TimeSeries.read(h1_file)
        l1_data = TimeSeries.read(l1_file)
    else:
        print('Fetching data from GWOSC...')
        start_time = GPS_START - 8  # 8 seconds before merger
        end_time = GPS_START + 4  # 4 seconds after merger
        h1_data = datasets.get_data('H1', start_time, end_time)
        l1_data = datasets.get_data('L1', start_time, end_time)
        # Save to disk
        h1_data.write(h1_file)
        l1_data.write(l1_file)
        print('Data saved to disk.')  

    # Validate data
    if h1_data.size == 0 or l1_data.size == 0:
        raise ValueError('Loaded data is empty.')

    # Plot strain vs time
    save_plot(h1_data.times, h1_data, 'Strain vs Time (H1)', 'GPS Time (s)', 'Strain', 'strain_h1_vs_time.png', 'H1 Strain')
    save_plot(l1_data.times, l1_data, 'Strain vs Time (L1)', 'GPS Time (s)', 'Strain', 'strain_l1_vs_time.png', 'L1 Strain')

except ValueError as ve:
    print(f'ValueError in Task 1: {ve}')
except Exception as e:
    print(f'Error in Task 1: {e}')

# Task 2: Data filtering
try:
    print('Whitening data...')
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    # Plot whitened data
    save_plot(h1_whitened.times, h1_whitened, 'Whitened Strain vs Time (H1)', 'GPS Time (s)', 'Whitened Strain', 'whitened_strain_h1.png', 'H1 Whitened')
    save_plot(l1_whitened.times, l1_whitened, 'Whitened Strain vs Time (L1)', 'GPS Time (s)', 'Whitened Strain', 'whitened_strain_l1.png', 'L1 Whitened')

    # Band-pass filter
    h1_filtered = h1_whitened.bandpass(30, 250)
    l1_filtered = l1_whitened.bandpass(30, 250)

    # Plot filtered data
    save_plot(h1_filtered.times, h1_filtered, 'Filtered Strain vs Time (H1)', 'GPS Time (s)', 'Filtered Strain', 'filtered_strain_h1.png', 'H1 Filtered')
    save_plot(l1_filtered.times, l1_filtered, 'Filtered Strain vs Time (L1)', 'GPS Time (s)', 'Filtered Strain', 'filtered_strain_l1.png', 'L1 Filtered')

except Exception as e:
    print(f'Error in Task 2: {e}')

# Task 3: Q-Transform
try:
    print('Creating Q-Transform plots...')
    plt.figure(figsize=(12, 6))
    plt.specgram(h1_filtered, NFFT=1024, Fs=1.0, noverlap=512)
    plt.colorbar(label='Energy')
    plt.clim(0, 25)
    plt.title('Q-Transform H1')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig('q_transform_h1.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.specgram(l1_filtered, NFFT=1024, Fs=1.0, noverlap=512)
    plt.colorbar(label='Energy')
    plt.clim(0, 25)
    plt.title('Q-Transform L1')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig('q_transform_l1.png')
    plt.close()
    print('Q-Transform plots saved.')

except Exception as e:
    print(f'Error in Task 3: {e}')

# Task 4: Data format conversion
try:
    print('Converting data formats for PyCBC...')
    h1_pycbc = h1_filtered.to_pycbc()
    l1_pycbc = l1_filtered.to_pycbc()
    print('Data formats converted.')

except Exception as e:
    print(f'Error in Task 4: {e}')

# Task 5: PyCBC template creation
try:
    print('Creating waveform templates...')
    masses = [10, 20, 30, 40]
    templates = []
    for mass in masses:
        # Use a valid approximant for waveform generation
        template = waveform.get_td_waveform(approximant='IMRPhenomD', mass1=mass, mass2=mass, spin1z=0, spin2z=0, f_lower=20, delta_t=1.0)
        if hasattr(template, 'duration') and template.duration > 0.2:
            # Pad or truncate to match data length
            template = template.crop(0, min(template.duration, h1_pycbc.duration))
            templates.append(template)
            save_plot(template.sample_times, template, f'Template {mass}M_sun vs Strain', 'GPS Time (s)', 'Amplitude', f'template_{mass}M_sun.png', f'Template {mass}M_sun')
        else:
            print(f'Warning: Template for mass {mass} is not a valid waveform.')

except Exception as e:
    print(f'Error in Task 5: {e}')

# Task 6: Calculating the Power Spectral Density
try:
    print('Calculating Power Spectral Density...')
    h1_psd = h1_pycbc.psd(4, 2)
    l1_psd = l1_pycbc.psd(4, 2)

    save_plot(h1_psd.frequency, h1_psd, 'Power Spectral Density (H1)', 'Frequency (Hz)', 'PSD', 'psd_h1_plot.png', 'H1 PSD')
    save_plot(l1_psd.frequency, l1_psd, 'Power Spectral Density (L1)', 'Frequency (Hz)', 'PSD', 'psd_l1_plot.png', 'L1 PSD')

except Exception as e:
    print(f'Error in Task 6: {e}')

# Task 7: Matched filtering
try:
    print('Performing matched filtering...')
    best_fit = {}  # Store best fit templates
    for template in templates:
        h1_snr = pycbc_filter.match(h1_pycbc, template)
        l1_snr = pycbc_filter.match(l1_pycbc, template)
        save_plot(h1_snr.sample_times, h1_snr, 'Matched Filtering SNR (H1)', 'GPS Time (s)', 'SNR', f'matched_filtering_h1_{template.mass1}.png', 'H1 SNR')
        save_plot(l1_snr.sample_times, l1_snr, 'Matched Filtering SNR (L1)', 'GPS Time (s)', 'SNR', f'matched_filtering_l1_{template.mass1}.png', 'L1 SNR')
        best_fit['H1'] = max(best_fit.get('H1', (0, None)), (h1_snr.max(), template.mass1))
        best_fit['L1'] = max(best_fit.get('L1', (0, None)), (l1_snr.max(), template.mass1))

    print(f'Best fit templates: {best_fit}')

except Exception as e:
    print(f'Error in Task 7: {e}')
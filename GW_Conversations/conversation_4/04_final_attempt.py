# filename: gw150914_analysis_with_checks.py

import os
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from gwosc.datasets import event_gps
from gwpy.signal import filter_design
import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform

# Task 1: Data fetching
try:
    print('Fetching event times for GW150914...')
    event_times = event_gps('GW150914')
    if isinstance(event_times, tuple) and len(event_times) == 2:
        gps_start, gps_end = event_times
        print(f'Start time: {gps_start}, End time: {gps_end}')
    else:
        raise ValueError('event_gps did not return a valid tuple of start and end times.')

    filename_h1 = 'gwosc_gw150914_h1.txt'
    filename_l1 = 'gwosc_gw150914_l1.txt'
    if os.path.exists(filename_h1) and os.path.exists(filename_l1):
        print('Loading existing data files...')
        h1_data = TimeSeries.read(filename_h1)
        l1_data = TimeSeries.read(filename_l1)
    else:
        print('Downloading data...')
        h1_data = TimeSeries.fetch_open_data('H1', gps_start - 8, gps_end + 4)
        l1_data = TimeSeries.fetch_open_data('L1', gps_start - 8, gps_end + 4)
        h1_data.write(filename_h1)
        l1_data.write(filename_l1)

    print('Plotting strain data...')
    plt.figure()
    h1_data.plot()
    plt.title('H1 Strain Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.savefig('h1_strain_plot.png')
    plt.close()

except Exception as e:
    print(f'Error in Task 1: {e}')

# Task 2: Data filtering
try:
    if 'h1_data' in locals():
        print('Whitening the data...')
        h1_whitened = h1_data.whiten()
        l1_whitened = l1_data.whiten()

        print('Plotting whitened data...')
        plt.figure()
        h1_whitened.plot()
        plt.title('Whitened H1 Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Whitened Strain')
        plt.savefig('h1_whitened_plot.png')
        plt.close()

        print('Applying band-pass filter...')
        sample_rate = h1_data.sample_rate
        h1_filtered = h1_whitened.bandpass(30, 250, sample_rate=sample_rate)
        l1_filtered = l1_whitened.bandpass(30, 250, sample_rate=sample_rate)
    else:
        print('H1 data is not defined. Skipping filtering.')

except Exception as e:
    print(f'Error in Task 2: {e}')

# Task 3: Q-Transform
try:
    if 'h1_filtered' in locals():
        print('Creating Q-Transform plots...')
        plt.figure()
        h1_filtered.q_transform().plot(normalize=True)
        plt.title('Q-Transform of H1 Filtered Data')
        plt.savefig('h1_q_transform.png')
        plt.close()
    else:
        print('H1 filtered data is not defined. Skipping Q-Transform.')

except Exception as e:
    print(f'Error in Task 3: {e}')

# Task 4: Template creation
try:
    if 'h1_data' in locals() and 'h1_filtered' in locals():
        print('Generating waveform templates...')
        masses = np.arange(10, 31, 1)
        templates = []
        for mass in masses:
            template = waveform.get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t=h1_data.delta_t)
            if template.duration > 0.2:
                templates.append(template)

        print('Scaling templates and plotting...')
        for i, template in enumerate(templates):
            if np.max(np.abs(template)) > 0:
                scaled_template = template / np.max(np.abs(template)) * np.max(np.abs(h1_filtered))
                plt.figure()
                plt.plot(h1_filtered.sample_times, h1_filtered, label='H1 Filtered Data')
                plt.plot(scaled_template.sample_times, scaled_template, label='Template')
                plt.title(f'Template Overlay for Mass {masses[i]}')
                plt.xlabel('Time (s)')
                plt.ylabel('Strain')
                plt.legend()
                plt.savefig(f'template_overlay_mass_{masses[i]}.png')
                plt.close()

        print('Saving combined H1 strain plot...')
        plt.figure()
        plt.plot(h1_filtered.sample_times, h1_filtered, label='H1 Filtered Data')
        plt.title('Combined H1 Strain Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.legend()
        plt.savefig('combined_h1_strain_plot.png')
        plt.close()
    else:
        print('H1 data or filtered data is not defined. Skipping template creation.')

except Exception as e:
    print(f'Error in Task 4: {e}')
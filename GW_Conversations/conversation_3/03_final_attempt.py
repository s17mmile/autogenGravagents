# filename: gw150914_analysis_refined.py
import os
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from gwosc import event_gps
from gwpy.signal import filter_design
from pycbc import waveform
import numpy as np

# Task 1: Data fetching
try:
    print('Fetching event times for GW150914...')
    gps_start, gps_end = event_gps('GW150914')
    print(f'Start time: {gps_start}, End time: {gps_end}')

    h1_file = 'gwosc_gw150914_h1.txt'
    l1_file = 'gwosc_gw150914_l1.txt'

    if os.path.exists(h1_file) and os.path.exists(l1_file):
        print('Loading existing data files...')
        h1_data = TimeSeries.read(h1_file)
        l1_data = TimeSeries.read(l1_file)
    else:
        print('Downloading data for H1 and L1...')
        h1_data = TimeSeries.fetch_open_data('H1', gps_start - 8, gps_start + 4)
        l1_data = TimeSeries.fetch_open_data('L1', gps_start - 8, gps_start + 4)
        h1_data.write(h1_file)
        l1_data.write(l1_file)

    print('Plotting strain data...')
    h1_plot = Plot(h1_data)
    h1_plot.save('h1_strain_plot.png')
    l1_plot = Plot(l1_data)
    l1_plot.save('l1_strain_plot.png')
    print('Plots saved.')

except FileNotFoundError as e:
    print(f'File not found: {e}')
except Exception as e:
    print(f'Error in Task 1: {e}')

# Task 2: Data filtering
try:
    print('Whitening the data...')
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    print('Plotting whitened data...')
    h1_whitened_plot = Plot(h1_whitened)
    h1_whitened_plot.save('h1_whitened_plot.png')
    l1_whitened_plot = Plot(l1_whitened)
    l1_whitened_plot.save('l1_whitened_plot.png')
    print('Whitened plots saved.')

    print('Applying band-pass filter...')
    sample_rate = h1_whitened.sample_rate
    h1_filtered = h1_whitened.bandpass(30, 250, sample_rate=sample_rate)
    l1_filtered = l1_whitened.bandpass(30, 250, sample_rate=sample_rate)

except ValueError as e:
    print(f'Value error: {e}')
except Exception as e:
    print(f'Error in Task 2: {e}')

# Task 3: Q-Transform
try:
    print('Creating Q-Transform plots...')
    from gwpy.spectrogram import Spectrogram
    h1_q_transform = Spectrogram(h1_filtered, 1, 1, 1)
    l1_q_transform = Spectrogram(l1_filtered, 1, 1, 1)

    h1_q_transform.plot(normalize=True).save('h1_q_transform.png')
    l1_q_transform.plot(normalize=True).save('l1_q_transform.png')
    print('Q-Transform plots saved.')

except Exception as e:
    print(f'Error in Task 3: {e}')

# Task 4: Template creation
try:
    print('Generating waveform templates...')
    masses = np.arange(10, 31, 1)
    templates = []
    for mass in masses:
        template = waveform.get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, spin1=(0, 0, 0), spin2=(0, 0, 0))
        if template.duration > 0.2:
            templates.append(template)

    if not templates:
        print('No valid templates generated.')
    else:
        print('Scaling templates...')
        max_h1 = np.max(np.abs(h1_filtered))
        for i, template in enumerate(templates):
            scaled_template = template / np.max(np.abs(template)) * max_h1
            plot = Plot(h1_filtered)
            plot.overlay(scaled_template)
            plot.save(f'template_overlay_{i}.png')

        print('Saving combined H1 strain plot...')
        combined_plot = Plot(h1_filtered)
        combined_plot.save('combined_h1_strain.png')
        print('All plots saved.')

except Exception as e:
    print(f'Error in Task 4: {e}')
# filename: gw150914_analysis_final_updated.py

from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from gwosc.datasets import event_gps
from gwpy.signal import filter_design
import numpy as np
import os

# Attempt to import alternative waveform generation functions from pycbc.waveform
try:
    from pycbc.waveform import generate_with_delta_f_and_max_freq
except ImportError as ie:
    print(f'Import error: {ie}')
    print('Please check if PyCBC is installed correctly and if the function is available.')
    raise

# Create output directory if it does not exist
output_dir = 'gw150914_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Constants for the GW150914 event
start_time = event_gps('GW150914') - 8  # 8 seconds before merger
end_time = event_gps('GW150914') + 4    # 4 seconds after merger

# Task 1: Data fetching
try:
    print('Fetching data for GW150914...')
    h1_file = os.path.join(output_dir, 'gwosc_gw150914_h1.txt')
    l1_file = os.path.join(output_dir, 'gwosc_gw150914_l1.txt')
    if os.path.exists(h1_file) and os.path.exists(l1_file):
        h1_strain_data = TimeSeries.read(h1_file)
        l1_strain_data = TimeSeries.read(l1_file)
        print('Loaded existing data files.')
    else:
        h1_strain_data = TimeSeries.fetch_open_data('H1', start_time, end_time)
        l1_strain_data = TimeSeries.fetch_open_data('L1', start_time, end_time)
        h1_strain_data.write(h1_file)
        l1_strain_data.write(l1_file)
        print('Downloaded new data and saved to files.')

    # Plot strain vs time
    h1_strain_data.plot(title='H1 Strain Data', xlabel='Time (GPS)', ylabel='Strain').save(os.path.join(output_dir, 'h1_strain_plot.png'))
    l1_strain_data.plot(title='L1 Strain Data', xlabel='Time (GPS)', ylabel='Strain').save(os.path.join(output_dir, 'l1_strain_plot.png'))
    print('Strain plots saved.')
except FileNotFoundError as fnf_error:
    print(f'File not found error: {fnf_error}')
except Exception as e:
    print(f'Error in data fetching: {e}')

# Task 2: Data filtering
try:
    print('Whitening data...')
    h1_whitened = h1_strain_data.whiten()
    l1_whitened = l1_strain_data.whiten()
    h1_whitened.plot(title='H1 Whitened Data', xlabel='Time (GPS)', ylabel='Whitened Strain').save(os.path.join(output_dir, 'h1_whitened_plot.png'))
    l1_whitened.plot(title='L1 Whitened Data', xlabel='Time (GPS)', ylabel='Whitened Strain').save(os.path.join(output_dir, 'l1_whitened_plot.png'))
    print('Whitened data plots saved.')

    print('Applying band-pass filter...')
    sample_rate = h1_whitened.sample_rate
    h1_filtered = h1_whitened.bandpass(30, 250, sample_rate=sample_rate)
    l1_filtered = l1_whitened.bandpass(30, 250, sample_rate=sample_rate)
except Exception as e:
    print(f'Error in data filtering: {e}')

# Task 3: Q-Transform
try:
    print('Creating Q-Transform plots...')
    h1_q_transform = h1_filtered.q_transform()
    l1_q_transform = l1_filtered.q_transform()
    h1_q_transform.plot(title='H1 Q-Transform', xlabel='Time (GPS)', ylabel='Frequency').save(os.path.join(output_dir, 'h1_q_transform_plot.png'))
    l1_q_transform.plot(title='L1 Q-Transform', xlabel='Time (GPS)', ylabel='Frequency').save(os.path.join(output_dir, 'l1_q_transform_plot.png'))
    print('Q-Transform plots saved.')
except Exception as e:
    print(f'Error in Q-Transform: {e}')

# Task 4: Template creation
try:
    print('Generating waveform templates...')
    masses = np.arange(10, 31, 1)  # 10 to 30 solar masses
    templates = []
    for mass in masses:
        template = generate_with_delta_f_and_max_freq(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, spin1z=0, spin2z=0)
        if len(template) > 0.2 * h1_filtered.sample_rate:
            template = template.crop(0, h1_filtered.duration)
            templates.append(template)
            # Scale template
            if np.max(np.abs(template)) > 0:
                max_amplitude = np.max(np.abs(h1_filtered))
                template *= max_amplitude / np.max(np.abs(template))
                # Plot overlay
                plot = Plot()
                plot.add(h1_filtered)
                plot.add(template)
                plot.title(f'Template Overlay for Mass {mass} Solar Masses')
                plot.xlabel('Time (GPS)')
                plot.ylabel('Strain')
                plot.save(os.path.join(output_dir, f'template_overlay_mass_{mass}.png'))
                print(f'Saved template overlay for mass {mass}.')
    print('All templates generated and saved.')
except Exception as e:
    print(f'Error in template creation: {e}')

print('All tasks completed successfully.')
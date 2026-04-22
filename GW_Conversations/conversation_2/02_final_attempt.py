# filename: gw150914_analysis_fixed_bandpass_v4.py
import logging
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import numpy as np
import matplotlib.pyplot as plt
from gwpy.signal.filter_design import bandpass
from pycbc import waveform

# Set up logging
logging.basicConfig(level=logging.INFO)

# Task 1: Data fetching
try:
    logging.info('Fetching event GPS times for GW150914.')
    gps_times = event_gps('GW150914')
    logging.info(f'GPS times returned: {gps_times}')
    if isinstance(gps_times, tuple):
        gps_start, gps_end = gps_times  # Expecting a tuple
    else:
        gps_start = gps_times  # Use single time if returned
        gps_end = gps_start + 1  # Set end time to 1 second later for fetching data
    logging.info(f'Start time: {gps_start}, End time: {gps_end}')

    logging.info('Downloading L1 and H1 strain data.')
    h1_data = TimeSeries.fetch_open_data('H1', gps_start - 8, gps_end + 4)
    l1_data = TimeSeries.fetch_open_data('L1', gps_start - 8, gps_end + 4)

    # Plot strain vs time for H1
    plt.figure(figsize=(10, 4))
    plt.plot(h1_data.times, h1_data, label='H1 Strain')
    plt.title('H1 Strain Data')
    plt.xlabel('Time (GPS)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid()
    plt.savefig('h1_strain.png')
    plt.close()
    logging.info('H1 strain plot saved.')
except Exception as e:
    logging.error(f'Error in data fetching: {e}')

# Task 2: Data filtering
try:
    logging.info('Whitening the data.')
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    # Plot whitened data
    plt.figure(figsize=(10, 4))
    plt.plot(h1_whitened.times, h1_whitened, label='H1 Whitened Strain')
    plt.title('H1 Whitened Strain Data')
    plt.xlabel('Time (GPS)')
    plt.ylabel('Whitened Strain')
    plt.legend()
    plt.grid()
    plt.savefig('h1_whitened.png')
    plt.close()
    logging.info('H1 whitened plot saved.')

    logging.info('Applying band-pass filter between 30 and 250 Hz using GWpy.')
    h1_filtered = bandpass(h1_whitened, 30, 250, sample_rate=h1_whitened.sample_rate)
    l1_filtered = bandpass(l1_whitened, 30, 250, sample_rate=l1_whitened.sample_rate)
except Exception as e:
    logging.error(f'Error in data filtering: {e}')

# Task 3: Q-Transform
try:
    logging.info('Creating Q-transform spectroscopy plot.')
    plt.figure(figsize=(10, 6))
    plt.specgram(h1_filtered, NFFT=1024, Fs=h1_filtered.sample_rate, Fc=0, noverlap=512)
    plt.colorbar(label='Normalized Energy')
    plt.title('Q-Transform of H1 Filtered Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (GPS)')
    plt.savefig('h1_q_transform.png')
    plt.close()
    logging.info('Q-transform plot saved.')
except Exception as e:
    logging.error(f'Error in Q-transform: {e}')

# Task 4: Template creation
try:
    logging.info('Generating waveform templates for H1 data.')
    masses = np.arange(10, 31, 1)
    templates = []
    for mass in masses:
        hp, hc = waveform.get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t=h1_filtered.delta_t)
        if hp.duration > 0.2:
            hp.resize(len(h1_filtered))
            max_hp = np.max(np.abs(hp))
            if max_hp > 0:
                hp /= max_hp  # Normalize
            templates.append(hp)
            # Plot overlay
            plt.figure(figsize=(10, 4))
            plt.plot(h1_filtered.times, h1_filtered, label='H1 Strain')
            plt.plot(hp.times, hp, label='Template (mass={mass})')
            plt.title(f'Template Overlay for Mass {mass}')
            plt.xlabel('Time (GPS)')
            plt.ylabel('Strain')
            plt.legend()
            plt.grid()
            plt.savefig(f'template_overlay_mass_{mass}.png')
            plt.close()
            logging.info(f'Template overlay for mass {mass} saved.')
    # Save combined H1 strain plot
    plt.figure(figsize=(10, 4))
    plt.plot(h1_filtered.times, h1_filtered, label='H1 Strain')
    plt.title('Combined H1 Strain Data')
    plt.xlabel('Time (GPS)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid()
    plt.savefig('combined_h1_strain.png')
    plt.close()
    logging.info('Combined H1 strain plot saved.')

    # Save templates for later analysis
    np.save('templates.npy', templates)
    logging.info('Templates saved for later analysis.')
except Exception as e:
    logging.error(f'Error in template creation: {e}')
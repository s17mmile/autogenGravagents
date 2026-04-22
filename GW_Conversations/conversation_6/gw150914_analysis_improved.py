# filename: gw150914_analysis_improved.py
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from pycbc import psd, filter, waveform
from gwpy.spectrogram import Spectrogram
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for the event
GPS_START = 1126059462  # GPS time for the merger
WINDOW_BEFORE = 8  # seconds before the merger
WINDOW_AFTER = 4  # seconds after the merger

# File names for the data
H1_FILE = 'gwosc_gw150914_h1.hdf5'
L1_FILE = 'gwosc_gw150914_l1.hdf5'

# Function to plot data
def plot_data(data, title, filename):
    plot = Plot(data)
    plot.title = title
    plot.xlabel = 'GPS Time (s)'
    plot.ylabel = 'Strain'
    plot.save(filename)
    logging.info(f'{title} saved as {filename}.')

# Task 1: Data fetching
try:
    # Check if data files exist
    if os.path.exists(H1_FILE) and os.path.exists(L1_FILE):
        logging.info('Loading data from disk...')
        h1_data = TimeSeries.read(H1_FILE)
        l1_data = TimeSeries.read(L1_FILE)
    else:
        logging.info('Downloading data...')
        h1_data = TimeSeries.fetch_open_data('H1', GPS_START - WINDOW_BEFORE, GPS_START + WINDOW_AFTER)
        l1_data = TimeSeries.fetch_open_data('L1', GPS_START - WINDOW_BEFORE, GPS_START + WINDOW_AFTER)
        # Save to disk
        h1_data.write(H1_FILE)
        l1_data.write(L1_FILE)

    # Validate data integrity
    if h1_data.is_empty() or l1_data.is_empty():
        raise ValueError('Loaded data is empty or invalid.')

    # Plot strain vs time
    plot_data(h1_data, 'H1 Strain Data', 'h1_strain_plot.png')
    plot_data(l1_data, 'L1 Strain Data', 'l1_strain_plot.png')

except Exception as e:
    logging.error(f'Error in data fetching: {e}')

# Task 2: Data filtering
try:
    # Whiten the data
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    # Plot whitened data
    plot_data(h1_whitened, 'Whitened H1 Data', 'h1_whitened_plot.png')
    plot_data(l1_whitened, 'Whitened L1 Data', 'l1_whitened_plot.png')

    # Apply band-pass filter
    h1_filtered = filter.bandpass(h1_whitened, 30, 250)
    l1_filtered = filter.bandpass(l1_whitened, 30, 250)

except Exception as e:
    logging.error(f'Error in data filtering: {e}')

# Task 3: Q-Transform
try:
    # Create Q-transform plots
    h1_spec = Spectrogram(h1_filtered, 1.0, 1.0)
    l1_spec = Spectrogram(l1_filtered, 1.0, 1.0)

    # Plot Q-transform for H1
    plot_data(h1_spec, 'Q-Transform H1', 'h1_q_transform_plot.png')
    # Plot Q-transform for L1
    plot_data(l1_spec, 'Q-Transform L1', 'l1_q_transform_plot.png')

except Exception as e:
    logging.error(f'Error in Q-Transform: {e}')
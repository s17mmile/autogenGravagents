# filename: gw150914_analysis_with_correct_bandpass.py
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from pycbc.filter import bandpass
from gwpy.spectrogram import Spectrogram
import logging

# Constants for the event
WINDOW_BEFORE = 8  # seconds before the merger
WINDOW_AFTER = 4  # seconds after the merger
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

# Task: Data filtering and Q-Transform
try:
    # Load data from disk
    h1_data = TimeSeries.read(H1_FILE)
    l1_data = TimeSeries.read(L1_FILE)

    # Whiten the data
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    # Plot whitened data
    plot_data(h1_whitened, 'Whitened H1 Data', 'h1_whitened_plot.png')
    plot_data(l1_whitened, 'Whitened L1 Data', 'l1_whitened_plot.png')

    # Apply band-pass filter using bandpass function
    h1_filtered = bandpass(h1_whitened, 30, 250, h1_whitened.sample_rate)
    l1_filtered = bandpass(l1_whitened, 30, 250, l1_whitened.sample_rate)

    # Create Q-transform plots
    h1_spec = Spectrogram(h1_filtered, 1.0, 1.0)
    l1_spec = Spectrogram(l1_filtered, 1.0, 1.0)

    # Plot Q-transform for H1
    plot_data(h1_spec, 'Q-Transform H1', 'h1_q_transform_plot.png')
    # Plot Q-transform for L1
    plot_data(l1_spec, 'Q-Transform L1', 'l1_q_transform_plot.png')

except Exception as e:
    logging.error(f'Error in data filtering or Q-Transform: {e}')
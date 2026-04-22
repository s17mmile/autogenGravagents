# filename: gw150914_analysis_final_corrected_v6.py
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import matplotlib.pyplot as plt

# Define constants for GPS time and offsets
GPS_TIME = 1126259462  # Merger time
OFFSET_BEFORE = 8  # seconds before
OFFSET_AFTER = 4   # seconds after

# Define file paths
parent_dir = os.path.dirname(os.getcwd())
file_h1 = os.path.join(parent_dir, 'gwosc_gw150914_h1.hdf5')
file_l1 = os.path.join(parent_dir, 'gwosc_gw150914_l1.hdf5')

# Helper function for plotting TimeSeries data

def save_plot(data, title, xlabel, ylabel, filename):
    try:
        plot = Plot(data)
        plot.title = title
        plot.xlabel = xlabel
        plot.ylabel = ylabel
        plot.save(filename)
        print(f'Saved plot: {filename}')
    except Exception as e:
        print(f'Error saving plot: {e}')

# Helper function for plotting Q-transform output
# Human correction here: color limit clipping. This is arguably quite a hard one to get right.
def save_q_transform_plot(q_data, title, filename):
    try:
        plot = q_data.plot(figsize=[8, 4], clim=(0, 25))
        ax = plot.gca()
        ax.set_title(title)
        ax.set_xscale('seconds')
        ax.set_yscale('log')
        ax.set_ylim(20, 500)
        ax.set_ylabel('Frequency [Hz]')
        ax.grid(True, axis='y', which='both')
        ax.colorbar(cmap='viridis', label='Normalized energy')
        plot.savefig(filename)
        plot.close()
        print(f'Saved Q-Transform plot: {filename}')
    except Exception as e:
        print(f'Error saving Q-Transform plot: {e}')

# Task 1: Data fetching
start_time = GPS_TIME - OFFSET_BEFORE
end_time = GPS_TIME + OFFSET_AFTER

try:
    if os.path.exists(file_h1) and os.path.exists(file_l1):
        print('Loading existing data files...')
        h1_data = TimeSeries.read(file_h1)
        l1_data = TimeSeries.read(file_l1)
    else:
        print('Downloading data...')
        h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time)
        l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time)
        h1_data.write(file_h1)
        l1_data.write(file_l1)
        print('Data saved to disk.')
except Exception as e:
    print(f'Error fetching data: {e}')
    exit(1)

# Plot strain data
save_plot(h1_data, 'H1 Strain Data', 'GPS Time (s)', 'Strain', os.path.join(parent_dir, 'h1_strain_plot.png'))
save_plot(l1_data, 'L1 Strain Data', 'GPS Time (s)', 'Strain', os.path.join(parent_dir, 'l1_strain_plot.png'))

# Task 2: Data filtering
print('Whitening data...')
try:
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()
except Exception as e:
    print(f'Error whitening data: {e}')
    exit(1)

# Plot whitened data
save_plot(h1_whitened, 'Whitened H1 Data', 'GPS Time (s)', 'Whitened Strain', os.path.join(parent_dir, 'h1_whitened_plot.png'))
save_plot(l1_whitened, 'Whitened L1 Data', 'GPS Time (s)', 'Whitened Strain', os.path.join(parent_dir, 'l1_whitened_plot.png'))

# Apply band-pass filter
print('Applying band-pass filter...')
h1_filtered = h1_whitened.bandpass(30, 250)
l1_filtered = l1_whitened.bandpass(30, 250)

# Plot filtered data
save_plot(h1_filtered, 'Filtered H1 Data', 'GPS Time (s)', 'Filtered Strain', os.path.join(parent_dir, 'h1_filtered_plot.png'))
save_plot(l1_filtered, 'Filtered L1 Data', 'GPS Time (s)', 'Filtered Strain', os.path.join(parent_dir, 'l1_filtered_plot.png'))

# Task 3: Q-Transform
print('Creating Q-Transform plots...')
try:
    q_h1 = h1_filtered.q_transform(outseg=(start_time, end_time))
    q_l1 = l1_filtered.q_transform(outseg=(start_time, end_time))
except Exception as e:
    print(f'Error creating Q-Transform: {e}')
    exit(1)

# Plot Q-Transform
save_q_transform_plot(q_h1, 'Q-Transform of H1 Data', os.path.join(parent_dir, 'h1_q_transform_plot.png'))
save_q_transform_plot(q_l1, 'Q-Transform of L1 Data', os.path.join(parent_dir, 'l1_q_transform_plot.png'))

print('Analysis complete! Check the parent directory for saved plots. Some plots may not have been saved successfully due to errors.')
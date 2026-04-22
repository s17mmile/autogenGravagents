# filename: fetch_gw150914_gps.py
import os
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

# Constants for the event
EVENT_NAME = 'GW150914'
WINDOW_BEFORE = 8  # seconds before the merger
WINDOW_AFTER = 4  # seconds after the merger

# Task: Fetch GPS time and download data
try:
    gps_time = event_gps(EVENT_NAME)
    print(f'GPS time for {EVENT_NAME}: {gps_time}')

    # Define the time window
    start_time = gps_time - WINDOW_BEFORE
    end_time = gps_time + WINDOW_AFTER

    # Check if data files exist
    H1_FILE = 'gwosc_gw150914_h1.hdf5'
    L1_FILE = 'gwosc_gw150914_l1.hdf5'

    if os.path.exists(H1_FILE) and os.path.exists(L1_FILE):
        print('Loading data from disk...')
        h1_data = TimeSeries.read(H1_FILE)
        l1_data = TimeSeries.read(L1_FILE)
    else:
        print('Downloading data...')
        h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time)
        l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time)
        # Save to disk
        h1_data.write(H1_FILE)
        l1_data.write(L1_FILE)

except Exception as e:
    print(f'Error in fetching GPS time or data: {e}')